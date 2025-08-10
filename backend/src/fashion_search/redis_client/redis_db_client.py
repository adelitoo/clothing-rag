import redis
import json
from typing import Dict, Any, Optional


class RedisDBClient:
    def __init__(self, host: str = "localhost", port: int = 6379):
        try:
            self.client = redis.Redis(host=host, port=port, decode_responses=True)
            self.client.ping()
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Could not connect to Redis: {e}")
            self.client = None

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None

        try:
            json_string = self.client.get(key)
            if json_string is None:
                return None
            return json.loads(json_string)
        except redis.exceptions.RedisError as e:
            print(f"Error getting JSON from Redis for key '{key}': {e}")
            try:
                key_type = self.client.type(key)
                print(
                    f"   - DEBUG: Expected key type 'string', but found '{key_type}'."
                )
            except redis.exceptions.RedisError as inner_e:
                print(f"   - DEBUG: Could not check type of key '{key}': {inner_e}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Data for key '{key}' is not valid JSON.")
            return None

    def set_json(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None):
        if not self.client:
            return

        try:
            json_string = json.dumps(data)
            self.client.set(key, json_string, ex=ttl)
        except redis.exceptions.RedisError as e:
            print(f"Error setting data in Redis for key '{key}': {e}")
        except TypeError:
            print(f"Error: Data for key '{key}' is not JSON serializable.")
            

    def set_hash(self, key: str, data: Dict[str, str]):
        if not self.client:
            return
        try:
            self.client.hset(key, mapping=data)
            print(f"✅ Successfully set hash for key '{key}' with {len(data)} fields.")
        except redis.exceptions.RedisError as e:
            print(f"Error setting hash in Redis for key '{key}': {e}")

    def get_hash(self, key: str) -> Dict[str, str]:
        if not self.client:
            return {}
        try:
            return self.client.hgetall(key)
        except redis.exceptions.RedisError as e:
            print(f"Error getting hash from Redis for key '{key}': {e}")
            return {}

