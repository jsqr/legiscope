import sys
import requests

def check_robots_txt(url):
    robots_url = f"{url.rstrip('/')}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            print(f"robots.txt exists at {robots_url}")
            print("\nContent:")
            print(response.text)
        else:
            print(f"robots.txt does not exist at {robots_url} (HTTP {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {robots_url}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_robots.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    check_robots_txt(url)
