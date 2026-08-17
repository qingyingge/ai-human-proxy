# 能工智人 (AI-Human Proxy)

> 对外提供兼容 OpenAI Chat Completions API 的本地 HTTP 服务，对内不含任何大语言模型——所有"推理"由真人通过 Web 界面逐字完成。

## 这是什么

一个反转了角色的 AI 网关：把 AI 聊天客户端（OpenAI SDK、opencode、Cursor 等）连接到本地服务，由**人类**实时接管模型回复。请求到来时，Web 界面弹出任务卡片，人打字，客户端就收到流式响应——适合需要精确控制输出、演示、测试、审校等场景。

```
┌──────────┐   OpenAI Chat API   ┌──────────────┐   WebSocket   ┌──────────┐
│ 客户端应用 │ ──────────────────> │ 能工智人服务端 │ <────────────> │ 人类控制台 │
│ openai SDK│ <────────────────── │ 127.0.0.1:11451│               │ (浏览器)  │
└──────────┘   流式 / 标准响应     └──────────────┘               └──────────┘
```

## 特性

- **协议对齐** OpenAI Chat Completions 2.3.0 规范：`stream`、`max_completion_tokens`、`stream_options.include_usage`、标准错误格式（四字段齐全）、流式 ErrorEvent
- **本地分词器** 加载 DeepSeek-V3 开源词表做精确 token 计数（不加载模型权重），UI 实时显示 prompt/completion token
- **人类控制台** 状态灯、轮次卡片、历史上下文折叠、markdown 渲染、特殊 token 一键插入、120s 超时保护
- **单文件后端** 依赖少、启动快，无数据库、无外部模型 API

## 快速开始

```bash
pip install -r requirements.txt
python main.py
# 浏览器打开 http://127.0.0.1:11451 即进入人工控制台
```

首次运行会从本地 `tokenizer/deepseek-v3/` 加载词表；若无本地词表则依次尝试 huggingface.co 与 hf-mirror.com 镜像。

## 接入你的客户端

任何 OpenAI 兼容客户端，指向 `http://127.0.0.1:11451/v1` 即可：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11451/v1", api_key="any")
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content, end="")
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI 兼容推理入口（`stream=false` 返回标准 JSON，`stream=true` 走 SSE） |
| `GET` | `/v1/models` | 模型列表（客户端探测用） |
| `WS` | `/ws` | 人类控制台双向通道 |
| `GET` | `/` | 人工控制台页面 |

### 人类控制台 WebSocket 协议

- 连接即收 `{"type":"ready","special_tokens":[...]}`（特殊 token 用于快捷插入）
- 有新请求时收到 `{"type":"new_request","request_id","model","max_tokens","prompt_tokens","messages"}`
- 人类只发两种消息：`{"type":"delta","content":"..."}` 增量、`{"type":"stop"}` 停止
- 输入过程中服务端推送实时 token 计数：`{"type":"usage","request_id","completion_tokens"}`

## 项目结构

```
├── main.py                  # FastAPI 服务端 (全部逻辑)
├── static/index.html        # 人工控制台 (零依赖原生 JS)
├── tokenizer/deepseek-v3/   # 本地 DeepSeek 词表
└── requirements.txt         # fastapi/uvicorn/pydantic/transformers
```

## 许可证

[MIT](./LICENSE) © qingyingge

本项目的代码由 opencode 与 DeepSeek 生成。