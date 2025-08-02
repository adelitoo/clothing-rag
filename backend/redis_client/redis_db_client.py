import redis
import json
from typing import Dict, Any, Optional

class RedisDBClient:
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.client.json().get(key)
            return result
        except redis.exceptions.RedisError as e:
            print(f"Error getting JSON from Redis for key '{key}': {e}")
            return None

    def set_json(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None):
        try:
            self.client.json().set(key, "$", data)
            if ttl:
                self.client.expire(key, ttl)
            print(f"Successfully cached data for key '{key}'")
        except redis.exceptions.RedisError as e:
            print(f"Error setting JSON in Redis for key '{key}': {e}")