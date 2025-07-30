import subprocess
import json
import webbrowser

command = [
    "python",
    "send_request.py",
    "--mode",
    "search",
    "--query",
    "white relaxed shoes",
    "--topk",
    "50",
]


def extract_json(output):
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        return None
    return output[start : end + 1]


def main():
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout

    json_str = extract_json(output)
    if not json_str:
        print("No JSON found in output")
        return

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        return

    for item in data.get("results", []):
        url = item.get("image")
        if url:
            print(f"Opening {url}")
            webbrowser.open(url)


if __name__ == "__main__":
    main()
