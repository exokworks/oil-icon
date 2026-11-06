#!/usr/bin/env python3
"""External-API fallback provider: OpenAI Images (gpt-image).

Used when the current agent has no callable image-generation tool and OpenAI
Images is the authorized external provider. For other APIs (Replicate, Fal,
Stability, …), call the selected provider directly instead.

  gen_image.py --prompt "<full sheet prompt>" --out raw/<style>.png

Reads OPENAI_API_KEY plus optional OPENAI_BASE_URL and OPENAI_USER_AGENT
from the environment. No third-party deps (stdlib only).
"""
import argparse, base64, binascii, json, os, sys, urllib.error, urllib.request


DEFAULT_MODEL = "gpt-image-2.6-sunburst"
FALLBACK_MODEL = "gpt-image-2.5-sunburst"
RETRYABLE_STATUS_CODES = {400, 403, 404, 422}
REQUEST_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    json.JSONDecodeError,
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    binascii.Error,
)


def error_reason(error):
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}"
    if isinstance(error, urllib.error.URLError):
        return "network or API request error"
    if isinstance(error, (TimeoutError, OSError)):
        return "network or file response error"
    return "invalid API response"


def uses_2_6(model):
    return model == "gpt-image-2.6" or model.startswith("gpt-image-2.6-")


def generate_image(model, prompt, size, base_url, key, user_agent):
    body = json.dumps({"model": model, "prompt": prompt, "size": size, "n": 1}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/images/generations",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": user_agent},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.load(response)
    encoded = data["data"][0]["b64_json"]
    image = base64.b64decode(encoded, validate=True)
    if not image:
        raise ValueError("empty image")
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--size", default="1024x1024")
    a = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY is not set")
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    user_agent = os.environ.get("OPENAI_USER_AGENT") or "Mozilla/5.0 (compatible; oil-icon/1.0)"

    try:
        image = generate_image(a.model, a.prompt, a.size, base_url, key, user_agent)
    except REQUEST_ERRORS as error:
        if (
            uses_2_6(a.model)
            and isinstance(error, urllib.error.HTTPError)
            and error.code in RETRYABLE_STATUS_CODES
        ):
            first_reason = error_reason(error)
            error.close()
            try:
                image = generate_image(FALLBACK_MODEL, a.prompt, a.size, base_url, key, user_agent)
            except REQUEST_ERRORS as fallback_error:
                fallback_reason = error_reason(fallback_error)
                if isinstance(fallback_error, urllib.error.HTTPError):
                    fallback_error.close()
                sys.exit(
                    f"Image generation failed with {a.model} ({first_reason}); "
                    f"automatic fallback {FALLBACK_MODEL} also failed "
                    f"({fallback_reason})."
                )
            print(
                f"{a.model} was rejected ({first_reason}); "
                f"used fallback {FALLBACK_MODEL}.",
                file=sys.stderr,
            )
        else:
            hint = ""
            if uses_2_6(a.model):
                hint = f" Automatic fallback to {FALLBACK_MODEL} was skipped."
            sys.exit(f"Image generation failed with {a.model} ({error_reason(error)}).{hint}")

    with open(a.out, "wb") as f:
        f.write(image)
    print(a.out)


if __name__ == "__main__":
    main()
