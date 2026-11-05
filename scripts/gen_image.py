#!/usr/bin/env python3
"""External-API fallback provider: OpenAI Images (gpt-image).

Used when the current agent has no callable image-generation tool and OpenAI
Images is the authorized external provider. For other APIs (Replicate, Fal,
Stability, …), call the selected provider directly instead.

  gen_image.py --prompt "<full sheet prompt>" --out raw/<style>.png

Reads OPENAI_API_KEY plus optional OPENAI_BASE_URL and OPENAI_USER_AGENT
from the environment. No third-party deps (stdlib only).
"""
import argparse, base64, json, os, sys, urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-image-1")
    ap.add_argument("--size", default="1024x1024")
    a = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY is not set")
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    user_agent = os.environ.get("OPENAI_USER_AGENT") or "Mozilla/5.0 (compatible; oil-icon/1.0)"

    body = json.dumps({"model": a.model, "prompt": a.prompt, "size": a.size, "n": 1}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/images/generations",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": user_agent},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    with open(a.out, "wb") as f:
        f.write(base64.b64decode(data["data"][0]["b64_json"]))
    print(a.out)


if __name__ == "__main__":
    main()
