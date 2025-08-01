import httpx
from typing import Dict, Any, Optional


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)

    async def run_pipeline(self, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            print(
                f"Sending POST request to {self.base_url}/pipeline/ with options: {options}"
            )
            response = await self.client.post("/pipeline/", json=options)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            print(
                f"Error response {e.response.status_code} while requesting {e.request.url!r}."
            )
            print(f"Details: {e.response.text}")
            return None

    async def search(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        payload = {"query": query, "top_k": top_k}
        try:
            response = await self.client.post("/search/", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            print(
                f"Error response {e.response.status_code} while requesting {e.request.url!r}."
            )
            print(f"Details: {e.response.text}")
            return None

    async def close(self):
        await self.client.aclose()
