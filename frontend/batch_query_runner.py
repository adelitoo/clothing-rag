import csv
import json
import subprocess
import sys
import time

QUERY_FILE = "./fashion_queries.csv"
GROUND_TRUTH_FILE = "./ground_truth.jsonl"
OUTPUT_FILE = "./combined_results.jsonl"
TOP_K = 10


def extract_json(output):
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        return None
    return output[start : end + 2]


ground_truth_dict = {}
with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        ground_truth_dict[entry["query"]] = entry["ground_truth"]

with (
    open(QUERY_FILE, newline="", encoding="utf-8") as csvfile,
    open(OUTPUT_FILE, "w", encoding="utf-8") as outfile,
):
    reader = csv.DictReader(csvfile)
    for row in reader:
        query = row["query"]
        print(f"🔍 Running query: {query}")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "send_request.py",
                    "--mode",
                    "search",
                    "--query",
                    query,
                    "--topk",
                    str(TOP_K),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            json_str = extract_json(result.stdout)
            if json_str:
                data = json.loads(json_str)
                data["ground_truth"] = ground_truth_dict.get(query, [])
                data["query"] = query
                outfile.write(json.dumps(data) + "\n")
            else:
                print(f"⚠️ No JSON found for query: {query}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error querying '{query}': {e}")

        time.sleep(0.5)  # To avoid overloading the server
