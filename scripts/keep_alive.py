"""
FinSecure - Automated Keep-Alive Utility.

Sends periodic HTTP requests to a deployed Streamlit app to prevent idle shutdown
or sleep states on free-tier platforms (Streamlit Community Cloud, Render, Hugging Face, etc.).
"""
import os
import sys
import time
import urllib.request
import urllib.error


def ping(url: str, timeout: int = 30) -> int:
    """Send an HTTP GET request to check the app status."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FinSecure-KeepAlive/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return 0


def main():
    target_url = os.environ.get("APP_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not target_url:
        print("Usage: python keep_alive.py <APP_URL> or set APP_URL environment variable.")
        print("Example: python keep_alive.py https://finsecure-fraud-detection.streamlit.app")
        sys.exit(1)

    target_url = target_url.strip()
    health_url = f"{target_url.rstrip('/')}/_stcore/health"

    print(f"Target URL: {target_url}")
    print(f"Health URL: {health_url}")

    status = ping(target_url)
    health_status = ping(health_url)

    print(f"App response status: {status}")
    print(f"Health response status: {health_status}")

    if 200 <= status < 400 or 200 <= health_status < 400:
        print("App is active and alive!")
        sys.exit(0)
    else:
        print("App returned non-2xx status. It might still be starting up.")
        sys.exit(1)


if __name__ == "__main__":
    main()
