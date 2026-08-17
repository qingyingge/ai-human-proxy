# -*- coding: utf-8 -*-
"""
能工智人 (AI-Human Proxy)
============================================================
对外提供兼容 OpenAI Chat Completion API 的本地 HTTP 服务,
对内不包含任何大语言模型 —— 所有"推理"由人类通过 Web UI 完成。

运行方式:
    pip install -r requirements.txt
    python main.py
浏览器打开 http://127.0.0.1:11451 即进入人工控制台。

协议要点:
    - 外部调用方: POST /v1/chat/completions (stream=True 走 SSE)
    - 人类控制台: 浏览器 WebSocket /ws, 仅发送两种消息
        {"type":"delta","content":"某字"}   增量
        {"type":"stop"}                     停止本轮生成
    - 人类 120 秒未停止 -> 超时错误
"""
import asyncio
import json
import os
import secrets
import time
from collections import deque
from typing import Any, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

HOST = "127.0.0.1"
PORT = 11451
TIMEOUT_SECONDS = 120          # 人类 120 秒未点击停止 -> 504
HEARTBEAT_SECONDS = 3          # 非流式: 每 3 秒发送一个换行符保持 TCP 活跃

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="能工智人 (AI-Human Proxy)")

# ---------------------------------------------------------------------------
# 分词器: 仅加载 DeepSeek-V3 开源词表用于 token 计数, 不加载模型权重。
# CPU 下约占 ~500MB 内存; 首次运行需联网下载词表文件到本地缓存。
# ---------------------------------------------------------------------------
TOKENIZER = None
LOCAL_TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer", "deepseek-v3")


def _load_tokenizer(source: str, local_only: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        source, trust_remote_code=True, local_files_only=local_only
    )


def _extract_special_tokens(tok) -> list:
    """从 tokenizer 提取模型真正使用的特殊 token (BOS/EOS/角色标记等)。"""
    seen, result = set(), []
    for v in tok.added_tokens_decoder.values():
        t = v.content
        if v.special and t and t not in seen and t.startswith("<"):
            seen.add(t)
            result.append(t)
    return result


try:
    # 1. 优先加载本地词表 (离线可用, 见 tokenizer/deepseek-v3/)
    if os.path.isdir(LOCAL_TOKENIZER_PATH):
        TOKENIZER = _load_tokenizer(LOCAL_TOKENIZER_PATH, local_only=True)
        print("[能工智人] DeepSeek-V3 tokenizer 加载完成 (本地)")
    else:
        raise FileNotFoundError("本地词表不存在")
except Exception as exc:
    # 2. 直连 huggingface.co
    print(f"[能工智人] 本地词表加载失败({exc}), 尝试联网加载...")
    try:
        TOKENIZER = _load_tokenizer("deepseek-ai/DeepSeek-V3")
        print("[能工智人] DeepSeek-V3 tokenizer 加载完成")
    except Exception as exc2:
        # 3. 回退 hf-mirror.com 镜像
        print(f"[能工智人] 直连加载失败({exc2}), 尝试 hf-mirror.com 镜像...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            TOKENIZER = _load_tokenizer("deepseek-ai/DeepSeek-V3")
            print("[能工智人] DeepSeek-V3 tokenizer 加载完成 (hf-mirror.com)")
        except Exception as exc3:  # 网络不可用 / 未安装 transformers 时的回退
            print(f"[能工智人] 警告: tokenizer 加载失败({exc3}), 回退为字符计数")

# 模型真正使用的特殊 token, 供 UI 一键插入 (BOS/EOS/角色标记等)
SPECIAL_TOKENS = _extract_special_tokens(TOKENIZER) if TOKENIZER else []


def count_tokens(text: str) -> int:
    """编码得到精确 token 数; 回退模式下按字符数近似。"""
    if TOKENIZER is None:
        return len(text)
    return len(TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# 请求模型 (对齐 OpenAI Chat Completion 协议)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""          # 可能是 str, 也可能是多模态数组(仅取 text)


class ChatCompletionStreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4"
    messages: List[ChatMessage] = []
    stream: bool = False
    max_tokens: Optional[int] = None              # 官方已标记 deprecated
    max_completion_tokens: Optional[int] = None   # 官方推荐字段, 兜底展示
    stream_options: Optional[ChatCompletionStreamOptions] = None
    tools: Optional[Any] = None   # 忽略扩展字段, 仅作占位


# ---------------------------------------------------------------------------
# 会话状态: 一个外部请求 = 一个 HumanSession, 排队等待人类逐个处理
# ---------------------------------------------------------------------------
class HumanSession:
    def __init__(self, request: ChatCompletionRequest):
        self.request_id = "chatcmpl-" + secrets.token_hex(8)  # 随机16位hex
        self.model = request.model
        self.messages = request.messages
        self.stream = request.stream
        # max_tokens 已废弃, 优先取 max_completion_tokens 用于 UI 展示提醒
        self.max_tokens = request.max_tokens or request.max_completion_tokens
        self.include_usage = bool(
            request.stream_options and request.stream_options.include_usage
        )
        self.created = int(time.time())
        self.deadline = time.time() + TIMEOUT_SECONDS

        self.delta_queue: asyncio.Queue = asyncio.Queue()  # 人类增量
        self.stop_event = asyncio.Event()                  # 人类点击停止
        self.active_event = asyncio.Event()                # 轮到本会话被服务
        self.parts: List[str] = []                         # 累计增量文本
        self.finished = False
        self.timed_out = False

        self.prompt_text = build_prompt_text(request.messages)
        self.prompt_tokens = count_tokens(self.prompt_text)


def build_prompt_text(messages: List[ChatMessage]) -> str:
    """将 messages 拼接为纯文本, 供 prompt token 计数。"""
    lines = []
    for m in messages:
        lines.append(f"{m.role}: {content_of(m)}")
    return "\n\n".join(lines)


def content_of(message: ChatMessage) -> str:
    """提取消息文本内容 (兼容 content 为数组的多模态格式)。"""
    c = message.content
    if isinstance(c, list):
        return "".join(part.get("text", "") for part in c if isinstance(part, dict))
    return str(c) if c is not None else ""


# ---------------------------------------------------------------------------
# 全局状态: 等待人类处理的会话队列 + 已连接的人类控制台
# ---------------------------------------------------------------------------
pending_sessions: deque = deque()
ui_clients: set = set()


def active_session() -> Optional[HumanSession]:
    """当前正等待人类服务的会话 (队首)。"""
    return pending_sessions[0] if pending_sessions else None


def session_payload(s: HumanSession) -> dict:
    """后端 -> UI 的 new_request 通知。"""
    return {
        "type": "new_request",
        "request_id": s.request_id,
        "model": s.model,
        "max_tokens": s.max_tokens,
        "prompt_tokens": s.prompt_tokens,
        "messages": [
            {"role": m.role, "content": content_of(m)} for m in s.messages
        ],
    }


async def broadcast(payload: dict):
    """向所有人类控制台广播。"""
    for ws in list(ui_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            ui_clients.discard(ws)


def mark_done(s: HumanSession):
    """会话终结: 出队, 提升下一个排队会话并通知 UI。"""
    if s.finished:
        return
    s.finished = True
    if s in pending_sessions:
        pending_sessions.remove(s)
    nxt = active_session()
    if nxt is not None and not nxt.active_event.is_set():
        nxt.active_event.set()
        asyncio.create_task(broadcast(session_payload(nxt)))


def error_body(message: str) -> dict:
    # 官方 Error 四字段全 required: type/message/param/code
    return {"error": {"message": message, "type": "server_error",
                      "param": None, "code": 504}}


def sse_error_event(message: str) -> str:
    # 官方 ErrorEvent: event: error + data: Error
    return ("event: error\n"
            + "data: " + json.dumps(error_body(message), ensure_ascii=False)
            + "\n\n")


def compute_usage(s: HumanSession) -> tuple:
    """仅在停止瞬间计数一次: completion_tokens = encode(全部增量文本)。"""
    content = "".join(s.parts)
    completion_tokens = count_tokens(content)
    return {
        "prompt_tokens": s.prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": s.prompt_tokens + completion_tokens,
    }, content


def sse_line(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


# ---------------------------------------------------------------------------
# 流式响应生成器 (Stream=True)
# ---------------------------------------------------------------------------
async def stream_response(s: HumanSession):
    try:
        # 等待轮到本会话 (排队中的请求先不发任何块)
        try:
            await asyncio.wait_for(
                s.active_event.wait(), timeout=max(0.0, s.deadline - time.time())
            )
        except asyncio.TimeoutError:
            s.timed_out = True
            yield sse_error_event("Inference timeout")
            yield "data: [DONE]\n\n"
            return

        # 1. 首块: 建立 assistant 角色 (官方示例 content 为空串)
        yield sse_line({
            "id": s.request_id,
            "object": "chat.completion.chunk",
            "created": s.created,
            "model": s.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                         "finish_reason": None}],
            **({"usage": None} if s.include_usage else {}),
        })

        # 2. 中间块: 原样透传 UI 增量 (不拆字, 不做 token 计数)
        # 同时监听"新增量"与"停止", 避免 stop 后仍阻塞在队列等待上
        stop_waiter = asyncio.create_task(s.stop_event.wait())
        while True:
            if s.stop_event.is_set() and s.delta_queue.empty():
                break
            remaining = s.deadline - time.time()
            if remaining <= 0:
                s.timed_out = True
                break
            get_task = asyncio.create_task(s.delta_queue.get())
            done, _ = await asyncio.wait(
                {get_task, stop_waiter}, timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:  # 人类 120 秒未停止
                get_task.cancel()
                s.timed_out = True
                break
            if stop_waiter in done:
                # 停止已到: 若队列还有未发完的增量, 先发完再结束
                get_task.cancel()
                if not s.delta_queue.empty():
                    delta = await s.delta_queue.get()
                    yield sse_line({
                        "id": s.request_id,
                        "object": "chat.completion.chunk",
                        "created": s.created,
                        "model": s.model,
                        "choices": [{"index": 0, "delta": {"content": delta},
                                     "finish_reason": None}],
                        **({"usage": None} if s.include_usage else {}),
                    })
                    continue
                break
            delta = get_task.result()
            yield sse_line({
                "id": s.request_id,
                "object": "chat.completion.chunk",
                "created": s.created,
                "model": s.model,
                "choices": [{"index": 0, "delta": {"content": delta},
                             "finish_reason": None}],
                **({"usage": None} if s.include_usage else {}),
            })

        if s.timed_out:
            # SSE 头部已发出无法改状态码, 以官方 ErrorEvent + [DONE] 结束
            yield sse_error_event("Inference timeout")
        else:
            # 3. 结束块: delta={} + finish_reason
            yield sse_line({
                "id": s.request_id,
                "object": "chat.completion.chunk",
                "created": s.created,
                "model": s.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                **({"usage": None} if s.include_usage else {}),
            })
            # 3b. usage 块: 仅当 stream_options.include_usage=true 时,
            #     在 [DONE] 前额外推送独立块, choices 恒为空数组
            if s.include_usage:
                usage, _ = compute_usage(s)
                yield sse_line({
                    "id": s.request_id,
                    "object": "chat.completion.chunk",
                    "created": s.created,
                    "model": s.model,
                    "choices": [],
                    "usage": usage,
                })

        # 4. 终止符
        yield "data: [DONE]\n\n"
    finally:
        mark_done(s)


# ---------------------------------------------------------------------------
# 非流式响应生成器 (Stream=False)
# ---------------------------------------------------------------------------
async def non_stream_response(s: HumanSession):
    try:
        # 等待轮到本会话, 期间持续发送换行心跳保持 TCP 活跃
        while not s.active_event.is_set():
            try:
                await asyncio.wait_for(s.active_event.wait(), HEARTBEAT_SECONDS)
                break
            except asyncio.TimeoutError:
                if time.time() > s.deadline:
                    s.timed_out = True
                    break
                yield "\n"

        # 阻塞等待人类点击停止, 每 3 秒一个换行心跳
        if not s.timed_out:
            while not s.stop_event.is_set():
                try:
                    await asyncio.wait_for(s.stop_event.wait(), HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    if time.time() > s.deadline:
                        s.timed_out = True
                        break
                    yield "\n"

        if s.timed_out:
            # 注: 心跳需要先行发出响应, 故超时时无法再改 HTTP 状态码,
            # 这里返回与规范一致的 504 错误 JSON 体。
            yield json.dumps(error_body("Inference timeout"), ensure_ascii=False)
        else:
            usage, content = compute_usage(s)
            yield json.dumps({
                "id": s.request_id,
                "object": "chat.completion",
                "created": s.created,
                "model": s.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content,
                                "refusal": None, "tool_calls": None},
                    "logprobs": None,
                    "finish_reason": "stop",
                }],
                "usage": usage,
            }, ensure_ascii=False)
    finally:
        mark_done(s)


# ---------------------------------------------------------------------------
# HTTP 端点: OpenAI 兼容入口
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    s = HumanSession(request)
    pending_sessions.append(s)
    if active_session() is s:
        s.active_event.set()
        await broadcast(session_payload(s))
    print(f"[能工智人] 收到推理请求 {s.request_id} "
          f"(stream={s.stream}, model={s.model})")

    if request.stream:
        return StreamingResponse(
            stream_response(s),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    return StreamingResponse(
        non_stream_response(s),
        media_type="application/json",
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# WebSocket 端点: 人类控制台
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def human_console(ws: WebSocket):
    await ws.accept()
    ui_clients.add(ws)
    # 连接确认, 消除"连接完成前广播已发出"的竞态
    await ws.send_json({"type": "ready", "special_tokens": SPECIAL_TOKENS})
    # 连接后立即同步当前正在等待的会话
    cur = active_session()
    if cur is not None:
        await ws.send_json(session_payload(cur))
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            s = active_session()
            if s is None or s.finished:
                continue
            if mtype == "delta":
                content = str(msg.get("content", ""))
                if content:
                    s.parts.append(content)
                    await s.delta_queue.put(content)
                    # 推送实时 token 计数, 供 UI 显示分词器的作用
                    await broadcast({
                        "type": "usage",
                        "request_id": s.request_id,
                        "completion_tokens": count_tokens("".join(s.parts)),
                    })
            elif mtype == "stop":
                s.stop_event.set()
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.discard(ws)


# ---------------------------------------------------------------------------
# Web UI 入口
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    """模型列表: opencode 等客户端启动时可能探测, 返回本地可用的模型。"""
    return {
        "object": "list",
        "data": [{
            "id": "deepseek-chat",
            "object": "model",
            "created": 0,
            "owned_by": "human",
        }],
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


def main():
    print(f"[能工智人] 服务运行在 http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()