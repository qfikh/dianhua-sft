# 垫话生成服务 API 文档
**服务地址:** `http://192.168.86.3:5011`
## 接口列表
### 1. 生成垫话
**POST** `/generate`

根据对话历史生成客服垫话（承接语）。

**请求参数（JSON）：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| dialog_history | string[] | 是 | 对话历史列表，每条为一轮对话，支持 `客户：`/`客服：` 或 `Client:`/`Service:` 前缀。自动截取最近10轮 |

**响应（JSON）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| code | number | 状态码，200 成功，500 失败 |
| message | string | 状态描述 |
| data | object | 业务数据 |
| data.dianhua | string | 生成的垫话内容，无效输入时返回空字符串 |
| data.latency | number | 调用耗时（秒），仅成功时返回 |

**curl 示例：**

```bash
curl -X POST http://192.168.86.3:5011/generate \
  -H "Content-Type: application/json" \
  -d '{"dialog_history": ["客服：您好，我是一汽大众的官方客服，看到您近期有在app上关注过我们的车，是吗？", "客户：诶，是的", "客服：嗯~请问您是在关注我们的哪款车型呢？", "客户：最新的车"]}'
```

**成功响应：**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "dianhua": "明白您的意思。",
    "latency": 0.213
  }
}
```

**对话为空响应：**

```json
{
  "code": 200,
  "message": "对话历史为空",
  "data": {"dianhua": ""}
}
```

**异常响应：**

```json
{
  "code": 500,
  "message": "Connection refused",
  "data": {"dianhua": ""}
}
```

---

### 2. 健康检查

**GET** `/health`

**响应：**

```json
{
  "code": 200,
  "message": "success",
  "data": {"status": "ok"}
}
```
