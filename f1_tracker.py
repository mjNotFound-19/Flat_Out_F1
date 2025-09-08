import requests
import pandas as pd
import time
import os
import re

def clean_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '', name).replace(" ", "_")

def get_latest_session():
    """Get info about the latest session (live or most recent)."""
    url = "https://api.openf1.org/v1/sessions?meeting_key=latest"
    resp = requests.get(url)
    resp.raise_for_status()
    session = resp.json()[0]
    return {
        "circuit": session["circuit_short_name"],
        "session": session["session_name"],
        "year": session["year"],
        "meeting_key": session["meeting_key"],
        "status": session["status"]   # e.g. "ongoing", "finished", "upcoming"
    }

def fetch_live_timing(meeting_key):
    url = f"https://api.openf1.org/v1/live_timing?meeting_key={meeting_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def save_to_csv(circuit, session, year, data):
    os.makedirs("F1_Data", exist_ok=True)
    filename = clean_filename(f"{circuit}_{session}_{year}.csv")
    filepath = os.path.join("F1_Data", filename)

    df = pd.DataFrame(data)

    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, mode="w")
    else:
        df.to_csv(filepath, index=False, mode="a", header=False)

    print(f"?? Saved {len(df)} rows  {filepath}")

def monitor(interval=30):
    """Runs 24x7 and collects data only when a race/session is live."""
    current_meeting = None

    while True:
        try:
            latest = get_latest_session()
            if latest["status"].lower() == "ongoing":
                # A new session started
                if current_meeting != latest["meeting_key"]:
                    print(f"?? New session detected: {latest['circuit']} - {latest['session']} ({latest['year']})")
                    current_meeting = latest["meeting_key"]

                data = fetch_live_timing(latest["meeting_key"])
                if data:
                    save_to_csv(latest["circuit"], latest["session"], latest["year"], data)
            else:
                print("? No live session right now. Waiting...")

        except Exception as e:
            print(f"?? Error: {e}")

        time.sleep(interval)

if __name__ == "__main__":
    monitor(interval=60)  # checks every 60 seconds

