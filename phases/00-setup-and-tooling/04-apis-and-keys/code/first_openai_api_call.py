import os
import json
import urllib.error
import urllib.request
from pathlib import Path

# Simple .env loader that looks in the current and parent directories for a .env file
def load_dotenv(override=True):
    for directory in [Path.cwd(), *Path.cwd().parents]:
        path = directory / ".env"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.removeprefix("export ").strip()
            if override or key not in os.environ:
                os.environ[key] = value.strip().strip("\"'")
        return


load_dotenv()

PROMPT = "What is a neural network in one sentence?"
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

def response_text(response):
    text = getattr(response, "output_text", None)
    if text:
        return text

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                return getattr(content, "text", "")
    return ""


def result_text(result):
    if result.get("output_text"):
        return result["output_text"]

    for item in result.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    return ""


def print_api_error(prefix, status, body):
    try:
        parsed = json.loads(body)
        message = parsed.get("error", {}).get("message", body)
    except json.JSONDecodeError:
        message = body

    print(f"{prefix} failed: HTTP {status}")
    print(f"Reason: {message[:500]}")
    if status == 429:
        print("Fix: check your OpenAI billing/quota or try again after the rate limit resets.")


def call_with_sdk():
    try:
        from openai import OpenAI
    except ImportError:
        print("Install the SDK: pip install openai")
        return
# 不用配置项，SDK会自动从环境变量读取OPENAI_API_KEY和OPENAI_MODEL
    client = OpenAI()
    try:
        response = client.responses.create(
            model=MODEL,
            input=PROMPT,
            max_output_tokens=256, # 调整输出长度限制以适应不同模型的默认值和能力
        )
        print(f"SDK response: {response_text(response)}")
        print(
            f"Tokens used: {response.usage.input_tokens} in, "
            f"{response.usage.output_tokens} out"
        )
    except Exception as exc:
        status = getattr(exc, "status_code", "unknown")
        message = getattr(exc, "message", str(exc))
        print(f"SDK request failed: HTTP {status}")
        print(f"Reason: {message}")
        if status == 429:
            print("Fix: check your OpenAI billing/quota or try again after the rate limit resets.")


def call_raw_http():
    # 直接使用HTTP请求调用API，展示底层细节
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable first")
        return

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = json.dumps({
        "model": MODEL,
        "input": PROMPT,
        "max_output_tokens": 256,
    }).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"Raw HTTP response: {result_text(result)}")
            print(
                f"Tokens used: {result['usage']['input_tokens']} in, "
                f"{result['usage']['output_tokens']} out"
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print_api_error("Raw HTTP request", exc.code, body)
    except urllib.error.URLError as exc:
        print(f"Raw HTTP request failed: {exc.reason}")


if __name__ == "__main__":
    print("=== API Calls ===\n")
    print("1. Using the SDK:")
    call_with_sdk()
    print("\n2. Using raw HTTP:")
    call_raw_http()
