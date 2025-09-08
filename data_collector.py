import requests
import pandas as pd
import time
import os
import re

# Clean filenames for saving
def clean_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '', name).replace(" ", "_")

# Get current session info
def get_current_session():
    url = "https://api.openf1.org/v1/sessions?meeting_key=latest"
    resp = requests.get(url)
    resp.raise_for_status()
    session = resp.json()[0]
    return (
        session["circuit_short_name"],  # e.g. Monza, Silverstone
        session["session_name"],        # e.g. Race, Qualifying
        session["year"],                # e.g. 2025
        session["meeting_key"]          # unique session ID
    )

# Fetch live timing data
def fetch_live_timing(meeting_key):
    url = f"https://api.openf1.org/v1/live_timing?meeting_key={meeting_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

# Save data into CSV per circuit + session + year
def save_to_csv(circuit_name, session_name, year, data):
    os.makedirs("F1_Data", exist_ok=True)

    filename = clean_filename(f"{circuit_name}_{session_name}_{year}.csv")
    filepath = os.path.join("F1_Data", filename)

    df = pd.DataFrame(data)

    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, mode="w")
    else:
        df.to_csv(filepath, index=False, mode="a", header=False)

    print(f"?? Saved {len(df)} rows  {filepath}")

# Poll live data
def poll_live(interval=10):
    circuit_name, session_name, year, meeting_key = get_current_session()
    print(f"?? Tracking {circuit_name} - {session_name} ({year})")

    while True:
        data = fetch_live_timing(meeting_key)
        if data:
            save_to_csv(circuit_name, session_name, year, data)
        time.sleep(interval)

if __name__ == "__main__":
    poll_live(interval=15)

