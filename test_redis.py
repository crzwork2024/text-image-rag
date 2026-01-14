import redis
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("测试 Redis 连接")
print("=" * 60)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

print(f"Host: {REDIS_HOST}")
print(f"Port: {REDIS_PORT}")
print(f"Password: '{REDIS_PASSWORD}' (长度: {len(REDIS_PASSWORD)})")
print(f"DB: {REDIS_DB}")
print()

# 测试1: 不带密码
print("测试1: 不传递密码参数")
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    r.ping()
    print("✅ 连接成功（无密码）")
except Exception as e:
    print(f"❌ 连接失败: {e}")

print()

# 测试2: 带空密码
print("测试2: 传递空字符串密码")
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password="", db=REDIS_DB)
    r.ping()
    print("✅ 连接成功（空密码）")
except Exception as e:
    print(f"❌ 连接失败: {e}")

print()

# 测试3: 带 None 密码
print("测试3: 传递 None 密码")
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=None, db=REDIS_DB)
    r.ping()
    print("✅ 连接成功（None密码）")
except Exception as e:
    print(f"❌ 连接失败: {e}")
