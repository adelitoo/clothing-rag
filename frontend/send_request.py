import asyncio
import argparse
import sys
import json
from client import ApiClient

API_SERVER_URL = "http://127.0.0.1:8000"


async def main():
    parser = argparse.ArgumentParser(
        description="Send requests to the fashion search server."
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "search"],
        required=True,
        help="Choose 'pipeline' to run data processing or 'search' to query items.",
    )

    parser.add_argument(
        "--query", type=str, default="", help="Text query for search mode."
    )
    parser.add_argument(
        "--topk", type=int, default=12, help="Number of results to return for search."
    )

    parser.add_argument(
        "--cleanup", action="store_true", help="Run the dataset cleanup step."
    )
    parser.add_argument(
        "--caption", action="store_true", help="Run the image captioning pipeline."
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Run the text embedding generation pipeline.",
    )
    parser.add_argument(
        "--insert-db",
        action="store_true",
        help="Insert embeddings into the vector database.",
    )

    args = parser.parse_args()
    client = ApiClient(base_url=API_SERVER_URL)

    try:
        if args.mode == "pipeline":
            options = {
                "run_cleanup": args.cleanup,
                "run_captioning": args.caption,
                "run_embeddings": args.embed,
                "run_db_insertion": args.insert_db,
            }

            if not any(options.values()):
                print(
                    "❌ Error: For pipeline mode, you must select at least one step: --cleanup, --caption, --embed, --insert-db"
                )
                sys.exit(1)

            print(f"🚀 Running pipeline with options: {options}")
            response = await client.run_pipeline(options)
            if response:
                print("✅ Pipeline request successful.")
                print(json.dumps(response, indent=2))
            else:
                print("❌ Pipeline request failed.")

        elif args.mode == "search":
            if not args.query:
                print("❌ Error: --query is required for search mode.")
                sys.exit(1)

            response = await client.search(args.query, args.topk)
            if response:
                print("✅ Search successful.")
                print(json.dumps(response, indent=2))
            else:
                print("❌ Search failed.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
