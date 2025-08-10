import json
from typing import List, Dict
from ..redis_client.redis_db_client import RedisDBClient
from ..core.config import settings

class ConversationManager:
    def __init__(self, redis_client: RedisDBClient):
        self.redis = redis_client
        print("✅ ConversationManager initialized.")

    def _get_key(self, session_id: str) -> str:
        return f"conversation:{session_id}"

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves conversation history from a Redis list."""
        if not self.redis.client:
            return []
            
        key = self._get_key(session_id)
        try:
            history_json = self.redis.client.lrange(key, 0, -1)
            history = [json.loads(item) for item in history_json]
            print(f"📚 Retrieved {len(history)} turns for session {session_id}")
            return history
        except Exception as e:
            print(f"⚠️ Could not retrieve history for session {session_id}: {e}")
            return []

    def add_turn(self, session_id: str, user_query: str, agent_summary: str):
        """Adds a user query and agent response summary to the history."""
        if not self.redis.client:
            return
            
        key = self._get_key(session_id)
        
        history_turns = [
            {"role": "User", "content": user_query},
            {"role": "Assistant", "content": agent_summary},
        ]
        
        try:
            pipeline = self.redis.client.pipeline()
            for turn in history_turns:
                pipeline.rpush(key, json.dumps(turn))
            
            pipeline.ltrim(key, -10, -1) 
            pipeline.expire(key, settings.CONVERSATION_TTL) 
            pipeline.execute()
            print(f"✍️ Added new turn to history for session {session_id}")

        except Exception as e:
            print(f"⚠️ Could not add to history for session {session_id}: {e}")