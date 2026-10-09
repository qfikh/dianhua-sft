import redis
import json

# Redis 配置
REDIS_HOST = "192.168.86.3"
REDIS_PORT = 6379
REDIS_PASSWORD = "Jgj_rmhpy@"

# 全局连接池（惰性初始化）
_pool = None
_client = None


def _get_client():
    global _pool, _client
    if _client is None:
        _pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
        _client = redis.Redis(connection_pool=_pool)
    return _client


def redis_get(key):
    """查询 Redis，返回字符串值，不存在返回 None"""
    try:
        value = _get_client().get(key)
        return value
    except Exception as e:
        print(f"[Redis GET] key={key} 失败: {e}")
        return None


def redis_set(key, value, ex=None):
    """写入 Redis，ex 为过期时间（秒）"""
    try:
        _get_client().set(key, value, ex=ex)
    except Exception as e:
        print(f"[Redis SET] key={key} 失败: {e}")


def redis_lpush(key, value, max_len=5, ex=1800):
    """向列表左侧推入值，超过 max_len 则截断，ex 为过期时间（秒）"""
    try:
        client = _get_client()
        client.lpush(key, value)
        client.ltrim(key, 0, max_len - 1)
        client.expire(key, ex)
    except Exception as e:
        print(f"[Redis LPUSH] key={key} 失败: {e}")


def redis_lrange(key, start=0, end=-1):
    """获取列表内容，返回字符串列表"""
    try:
        return _get_client().lrange(key, start, end)
    except Exception as e:
        print(f"[Redis LRANGE] key={key} 失败: {e}")
        return []


if __name__ == "__main__":
    # 测试连接
    redis_set("test_key", "hello_redis", ex=60)
    print(f"GET test_key: {redis_get('test_key')}")

    # 测试列表操作
    test_list_key = "test_list"
    for item in ["嗯嗯。", "好的。", "收到。"]:
        redis_lpush(test_list_key, item, max_len=5, ex=60)
    print(f"LRANGE test_list: {redis_lrange(test_list_key)}")
