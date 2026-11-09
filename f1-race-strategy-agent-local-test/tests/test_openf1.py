
import requests

url = "https://api.openf1.org/v1/sessions"

response = requests.get(
    url,
    params={
        "year": 2023,
        "country_name": "Bahrain",
        "session_name": "Race",
    },
    timeout=30,
)

response.raise_for_status()

for session in response.json():
    print("Session key:", session["session_key"])
    print("Session:", session["session_name"])
    print("Location:", session["location"])

