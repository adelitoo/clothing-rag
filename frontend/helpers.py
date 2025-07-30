import subprocess
import json
import sys


def extract_json(output):
    """Extracts JSON string from command output"""
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        return None
    return output[start : end + 1]


def run_query(query, topk):
    """Execute search command and return parsed JSON"""
    command = [
        sys.executable,
        "send_request.py",
        "--mode",
        "search",
        "--query",
        query,
        "--topk",
        str(topk),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        json_str = extract_json(result.stdout)
        return json.loads(json_str) if json_str else None
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error executing command: {str(e)}")
        return None
