import requests
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SearchClient:
    def __init__(self, systems_config: Dict[str, Dict[str, Any]], timeout: int = 30):
        self.systems = systems_config
        self.timeout = timeout
        logging.info(f"SearchClient initialized for systems: {list(self.systems.keys())}")

    def get_search_results(self, system_name: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        if system_name not in self.systems:
            logging.error(f"System '{system_name}' not found in configuration.")
            return []

        api_url = self.systems[system_name]['url']
        payload = {"query": query, "top_k": top_k}

        try:
            response = requests.post(api_url, json=payload, timeout=self.timeout)
            response.raise_for_status() 
            
            results = response.json().get("results", [])
            for item in results:
                if 'article_id' in item:
                    item['article_id'] = int(item['article_id'])
            return results

        except requests.exceptions.RequestException as e:
            logging.warning(f"API request failed for query '{query}' on system '{system_name}': {e}")
            return []