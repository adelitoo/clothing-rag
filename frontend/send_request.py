import json
import asyncio
import httpx
import sys
import time
import argparse
from yaspin import yaspin

SERVER = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 600


async def send_request_with_timeout(endpoint: str, json_data: dict):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            response = await asyncio.wait_for(
                client.post(f"{SERVER}/{endpoint}/", json=json_data),
                timeout=TIMEOUT_SECONDS,
            )
            return response
        except (httpx.RequestError, asyncio.TimeoutError):
            return None


def run_pipeline(options):
    with yaspin(text="Sending pipeline request...", color="cyan") as spinner:
        response = asyncio.run(send_request_with_timeout("pipeline", options))
        if response:
            spinner.ok("✅")
            print("✔️ Pipeline response received:", response.status_code)
            print("📝 Pipeline response content:", response.text)
        else:
            spinner.fail("⏳ Pipeline request timed out or failed.")
            print("No pipeline response received within timeout.")


def run_search(query: str, top_k: int = 10):
    payload = {"query": query, "top_k": top_k}
    with yaspin(text=f"Searching for '{query}'...", color="green") as spinner:
        response = asyncio.run(send_request_with_timeout("search", payload))
        if response:
            spinner.ok("✅")
            print(f"✔️ Search response received: {response.status_code}")
            print("📝 Search results:", json.dumps(response.json(), indent=4))
        else:
            spinner.fail("⏳ Search request timed out or failed.")
            print("No search response received within timeout.")


def main():
    parser = argparse.ArgumentParser(description="Send requests to your server.")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "search"],
        required=True,
        help="Mode to run: pipeline or search",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Search query string (only for search mode)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of search results to return (only for search mode)",
    )

    args = parser.parse_args()

    if args.mode == "pipeline":
        options = {
            "run_cleaning": False,
            "run_captioning": False,
            "run_embedding": False,
            "run_db_insertion": True,
        }
        run_pipeline(options)

    elif args.mode == "search":
        if not args.query:
            print("Error: --query is required in search mode.")
            sys.exit(1)
        run_search(args.query, args.topk)


if __name__ == "__main__":
    main()
    time.sleep(2)
    sys.exit(0)
