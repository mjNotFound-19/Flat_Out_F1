import fastf1 as ff1
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import os
import plotly.express as px
import difflib
import requests

# -----------------------
# Cache setup
# -----------------------
CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

# -----------------------
# Circuit-specific static data
# -----------------------
CIRCUIT_FEATURES = {
    "Monza": {"Length":5.793, "Corners":11, "OvertakingDifficulty":0.2},
    "Silverstone": {"Length":5.891, "Corners":18, "OvertakingDifficulty":0.3},
    "Suzuka": {"Length":5.807, "Corners":18, "OvertakingDifficulty":0.35},
    "Azerbaijan": {"Length":6.003, "Corners":20, "OvertakingDifficulty":0.45},
    # Add more circuits as needed
}

# -----------------------
# GP Resolution
# -----------------------
def resolve_gp_input(year, gp_name=None, round_num=None):
    schedule = ff1.get_event_schedule(year)
    if gp_name:
        gps = schedule['EventName'].tolist()
        matches = [g for g in gps if gp_name.lower() in g.lower()]
        if matches:
            return matches[0]
        matches = difflib.get_close_matches(gp_name, gps, n=1, cutoff=0.3)
        if matches:
            return matches[0]
        raise ValueError(f"No GP found matching '{gp_name}' in {year}.")
    if round_num:
        if round_num <= len(schedule):
            return schedule.iloc[round_num-1]['EventName']
        else:
            raise ValueError(f"Round {round_num} not found in {year}.")
    raise ValueError("Must provide either gp_name or round_num")

# -----------------------
# Fetch historical race data
# -----------------------
def fetch_race_data(year, gp_name):
    try:
        session = ff1.get_session(year, gp_name, 'R')
        session.load(laps=True)
        laps = session.laps
        df = laps[['Driver','LapNumber','Position','Stint','Compound','TyreLife','LapTime','PitInTime']].copy()
        df['LapTime'] = df['LapTime'].dt.total_seconds()
        df['PitStops'] = df['PitInTime'].notnull().astype(int)
        return df
    except:
        return pd.DataFrame()

# -----------------------
# Fetch qualifying data
# -----------------------
def fetch_qualifying_data(year, gp_name):
    try:
        session = ff1.get_session(year, gp_name, 'Q')
        session.load(laps=True)
        q = session.laps[['Driver','LapTime']].copy()
        q['LapTime'] = q['LapTime'].dt.total_seconds()
        q['GridPosition'] = q.groupby('Driver')['LapTime'].transform('min').rank(method='first')
        return q.groupby('Driver').agg({'GridPosition':'first','LapTime':'min'}).reset_index()
    except:
        return pd.DataFrame()

# -----------------------
# Fetch championship points (live Ergast or fallback)
# -----------------------
def fetch_championship_points(year):
    driver_points = {}
    constructor_points = {}
    try:
        url = f"https://ergast.com/api/f1/{year}/driverStandings.json"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        standings_list = r.json()['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        for item in standings_list:
            driver = item['Driver']['code']
            pts = float(item['points'])
            driver_points[driver] = pts
            constructor = item['Constructors'][0]['name']
            constructor_points[constructor] = constructor_points.get(constructor,0) + pts
    except Exception as e:
        print(f"Warning: Could not fetch live standings ({e}). Using zeros.")
        driver_points = {}
        constructor_points = {}
    # Optional local CSV fallback
    csv_file = f"points_{year}.csv"
    if not driver_points and os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        driver_points = dict(zip(df['Driver'], df['Points']))
        constructor_points = df.groupby('Constructor')['Points'].sum().to_dict()
    return driver_points, constructor_points

# -----------------------
# Compute historical averages
# -----------------------
def compute_driver_historical_averages(historical_dfs, recency_weights=[0.6,0.3,0.1]):
    hist_avg = []
    for i, df in enumerate(historical_dfs):
        if df.empty:
            continue
        weight = recency_weights[i] if i<len(recency_weights) else recency_weights[-1]
        grouped = df.groupby('Driver').agg({
            'LapTime':'mean',
            'Stint':'mean',
            'PitStops':'sum'
        }).reset_index()
        grouped[['LapTime','Stint','PitStops']] *= weight
        hist_avg.append(grouped[['Driver','LapTime','Stint','PitStops']])
    if not hist_avg:
        return pd.DataFrame(columns=['Driver','HistAvgLapTime','HistStints','HistPitStops'])
    combined = pd.concat(hist_avg).groupby('Driver').sum().reset_index()
    combined.rename(columns={'LapTime':'HistAvgLapTime','Stint':'HistStints','PitStops':'HistPitStops'}, inplace=True)
    return combined

# -----------------------
# Compute driver form trend
# -----------------------
def compute_driver_form_trend(historical_dfs):
    form_dict = {}
    for df in historical_dfs[-6:]:
        if df.empty:
            continue
        grouped = df.groupby('Driver')['LapTime'].mean()
        for driver, lap_time in grouped.items():
            if driver not in form_dict:
                form_dict[driver] = []
            form_dict[driver].append(lap_time)
    trend_data = {}
    for driver, laps in form_dict.items():
        if len(laps) >= 6:
            recent_avg = np.mean(laps[-3:])
            previous_avg = np.mean(laps[:3])
            trend_data[driver] = max(previous_avg - recent_avg, 0)
        else:
            trend_data[driver] = 0
    return trend_data

# -----------------------
# Prepare dataset
# -----------------------
def prepare_dataset(qual_df, hist_df, driver_points, constructor_points, gp_name, driver_trend=None):
    df = pd.merge(qual_df, hist_df, on='Driver', how='left')
    for col in ['HistAvgLapTime','HistStints','HistPitStops']:
        df[col] = df.get(col, 0)
        df[col] = df[col].fillna(df[col].mean() if df[col].notnull().any() else 0)
    df['DriverPoints'] = df['Driver'].map(driver_points).fillna(0)
    # Map constructor points
    # Simplified: assign constructor points based on driver mapping
    constructor_map = {d: c for c in df['Driver'] for c in constructor_points.keys()}
    df['ConstructorPoints'] = df['Driver'].map(lambda d: constructor_points.get(constructor_map.get(d,''),0)).fillna(0)
    df['DriverFormBoost'] = df['Driver'].map(driver_trend).fillna(0) if driver_trend else 0
    circuit = CIRCUIT_FEATURES.get(gp_name.split()[0], {'Length':5.5,'Corners':12,'OvertakingDifficulty':0.3})
    for k,v in circuit.items():
        df[k] = v
    return df

# -----------------------
# Train model (XGBoost)
# -----------------------
def train_model(df):
    features = ['GridPosition','HistAvgLapTime','HistPitStops','HistStints',
                'DriverPoints','ConstructorPoints','Length','Corners','OvertakingDifficulty',
                'DriverFormBoost']
    df['FinishPosition'] = df['HistAvgLapTime'].rank(method='first') + df['GridPosition']*0.5
    X = df[features]
    y = df['FinishPosition']
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=4, learning_rate=0.1)
    model.fit(X, y)
    return model

# -----------------------
# Monte Carlo simulation
# -----------------------
def simulate_race(df, model, simulations=3000):
    features = ['GridPosition','HistAvgLapTime','HistPitStops','HistStints',
                'DriverPoints','ConstructorPoints','Length','Corners','OvertakingDifficulty',
                'DriverFormBoost']
    driver_positions = {driver: [] for driver in df['Driver']}
    for _ in range(simulations):
        lap_mod = np.random.normal(0,0.3,len(df))
        tire_deg = np.random.uniform(0,0.5,len(df))
        preds = model.predict(df[features]) + lap_mod + tire_deg
        final_order = pd.Series(preds,index=df['Driver']).sort_values().index
        for pos, driver in enumerate(final_order,1):
            driver_positions[driver].append(pos)
    summary = []
    for driver, positions in driver_positions.items():
        summary.append({
            'Driver':driver,
            'PredictedPosition':np.mean(positions),
            'ProbWin':np.mean([p==1 for p in positions])
        })
    return pd.DataFrame(summary).sort_values('PredictedPosition')

# -----------------------
# Visualization
# -----------------------
def save_interactive_html(df, filename):
    fig = px.bar(df, x='Driver', y='PredictedPosition', hover_data=['ProbWin'], title='F1 Race Prediction')
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.write_html(filename)
    print(f"Interactive HTML saved as {filename}")

# -----------------------
# Main
# -----------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gp', type=str)
    parser.add_argument('--round_num', type=int)
    parser.add_argument('--year', type=int, default=pd.Timestamp.now().year)
    parser.add_argument('--seasons', nargs='+', type=int, default=[2022,2023,2024])
    parser.add_argument('--qual_csv', type=str, help="CSV file for qualifying override")
    args = parser.parse_args()

    gp_name = resolve_gp_input(args.year, gp_name=args.gp, round_num=(args.round_num+1))
    print(f"Predicting for GP: {gp_name} ({args.year})")

    historical_dfs = [fetch_race_data(y, gp_name) for y in args.seasons]

    if args.qual_csv and os.path.exists(args.qual_csv):
        qual_df = pd.read_csv(args.qual_csv)
        print(f"Using qualifying data from {args.qual_csv}")
    else:
        qual_df = fetch_qualifying_data(args.year, gp_name)

    driver_points, constructor_points = fetch_championship_points(args.year)
    hist_avgs = compute_driver_historical_averages(historical_dfs)
    driver_trend = compute_driver_form_trend(historical_dfs)

    train_df = prepare_dataset(qual_df, hist_avgs, driver_points, constructor_points, gp_name, driver_trend)

    model = train_model(train_df)
    predictions = simulate_race(train_df, model, simulations=3000)

    CSV_DIR = os.path.join("predictions", "csv")
    HTML_DIR = os.path.join("predictions", "html")
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)

    csv_file = os.path.join(CSV_DIR, f'{args.year}_predicted_{gp_name.replace(" ","_")}.csv')
    html_file = os.path.join(HTML_DIR, f'{args.year}_predicted_{gp_name.replace(" ","_")}.html')

    predictions.to_csv(csv_file, index=False)
    save_interactive_html(predictions, html_file)

    print(f"CSV saved to {csv_file}")
    print(f"Interactive HTML saved to {html_file}")

