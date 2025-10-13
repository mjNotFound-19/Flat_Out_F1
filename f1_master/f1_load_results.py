import os
import argparse
import pandas as pd
import fastf1

OUTPUT_DIR = "actual_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_results_from_fastf1(year: int, round_num: int):
    """Fetch official race results from FastF1."""
    print(f"?? Loading race results for Round {round_num} ({year}) via FastF1...")
    session = fastf1.get_session(year, round_num, 'R')

    try:
        session.load()
    except Exception as e:
        print(f"?? Could not load FastF1 session for Round {round_num}: {e}")
        return None, None

    results = session.results
    race_name = session.event.EventName.replace(" ", "_")

    df = pd.DataFrame({
        "DriverNumber": results["DriverNumber"].astype(str),
        "Driver": results["Abbreviation"],
        "Constructor": results["TeamName"],
        "FinishPosition": results["Position"].astype(int),
        "Grid": results["GridPosition"].astype(int),
        "Points": results["Points"].astype(float)
    })

    df.sort_values("FinishPosition", inplace=True)
    return df, race_name


def save_results_csv(df, year, race_name):
    """Save the race results to CSV."""
    file_path = os.path.join(OUTPUT_DIR, f"{year}_{race_name}_results.csv")
    df.to_csv(file_path, index=False)
    print(f"? Results saved  {file_path}")


def parse_rounds(round_str: str):
    """Parse ranges like '1-5' or lists like '1 3 5'."""
    rounds = []
    for part in round_str.split():
        if "-" in part:
            start, end = part.split("-")
            rounds.extend(range(int(start), int(end) + 1))
        else:
            rounds.append(int(part))
    return sorted(set(rounds))


def main():
    parser = argparse.ArgumentParser(description="Load official F1 results using FastF1.")
    parser.add_argument("--year", type=int, default=2025, help="Season year (e.g. 2025)")
    parser.add_argument("--round_num", type=str, required=True, help="Round(s) - e.g. '1-18' or '2 5 7'")
    args = parser.parse_args()

    rounds = parse_rounds(args.round_num)

    print(f"?? Loading results for rounds: {rounds}")
    for rnd in rounds:
        df, race_name = fetch_results_from_fastf1(args.year, rnd)
        if df is not None:
            save_results_csv(df, args.year, race_name)


if __name__ == "__main__":
    main()

