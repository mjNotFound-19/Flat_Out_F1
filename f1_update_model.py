import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from datetime import datetime

# -----------------------
# Neural net class - same as in predictor
# -----------------------
class F1PredictorNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------
# Load models
# -----------------------
def load_xgb_model(path):
    model = xgb.XGBRegressor()
    model.load_model(path)
    print(f"Loaded XGBoost model from {path}")
    return model

def load_nn_model(path, input_dim):
    checkpoint = torch.load(path)
    model = F1PredictorNN(input_dim)
    model.load_state_dict(checkpoint['model_state'])
    scaler = checkpoint.get('scaler', None)
    print(f"Loaded NN model (input_dim={input_dim}) from {path}")
    return model, scaler

# -----------------------
# Evaluate past learning log
# -----------------------
def has_already_learned(log_path, year, gp_name):
    """Check a log file if this GP has already been used to update."""
    if not os.path.exists(log_path):
        return False
    df = pd.read_csv(log_path)
    matches = df[(df['Year'] == year) & (df['GP'] == gp_name)]
    return not matches.empty

# -----------------------
# Log update activity
# -----------------------
def log_update(log_path, year, gp_name, model_type, mae_before, mae_after):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    entry = pd.DataFrame({
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Year": [year],
        "GP": [gp_name],
        "ModelType": [model_type],
        "MAE_before": [mae_before],
        "MAE_after": [mae_after]
    })
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, entry], ignore_index=True)
    else:
        df = entry
    df.to_csv(log_path, index=False)
    print(f"Logged update for {gp_name} {year} in {log_path}")

# -----------------------
# Retraining logic
# -----------------------
def retrain_xgb(model, feat_df, result_df, alpha=1.0):
    # feat_df: includes features + perhaps predictions; result_df has Driver & Position
    features = ['GridPosition','HistAvgLapTime','HistPitStops','HistStints',
                'DriverPoints','ConstructorPoints','Length','Corners','OvertakingDifficulty','DriverFormBoost']
    df = pd.merge(feat_df, result_df, on='Driver', how='inner')
    # clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['Position'])
    df[features] = df[features].fillna(df[features].mean())
    X = df[features]
    y = df['Position']
    # we can also blend predictions with true label to avoid overfitting
    if alpha < 1.0:
        pred0 = model.predict(X)
        target = (1 - alpha) * pred0 + alpha * y
    else:
        target = y
    model.fit(X, target, xgb_model=model.get_booster(), verbose=False)
    return model

def retrain_nn(model, scaler, feat_df, result_df, alpha=1.0, lr=1e-3, epochs=100):
    features = ['GridPosition','HistAvgLapTime','HistPitStops','HistStints',
                'DriverPoints','ConstructorPoints','Length','Corners','OvertakingDifficulty','DriverFormBoost']
    df = pd.merge(feat_df, result_df, on='Driver', how='inner')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['Position'])
    df[features] = df[features].fillna(df[features].mean())
    X = df[features].values
    y = df['Position'].values.reshape(-1,1)

    # scale features
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    # optionally blend initial predictions
    with torch.no_grad():
        pred0 = model(X_t).cpu().numpy().reshape(-1,1)
    target = (1 - alpha) * pred0 + alpha * y

    target_t = torch.tensor(target, dtype=torch.float32).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, target_t)
        loss.backward()
        optimizer.step()

    # compute MAE
    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy().reshape(-1)
    mae_after = mean_absolute_error(y.reshape(-1), preds)
    return model, scaler, mae_after

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--result_csv", required=True)
    parser.add_argument("--model_type", choices=["xgb", "nn"], default="xgb")
    parser.add_argument("--model_path", default="models/f1_model")
    parser.add_argument("--gp", required=True, help="GP name")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--log", default="update_log.csv")
    parser.add_argument("--alpha", type=float, default=1.0, help="Blend factor: 1.0 = full learning, <1 = partial update")
    parser.add_argument("--epochs", type=int, default=100, help="NN epochs, if using nn")
    args = parser.parse_args()

    # Check if already learned
    if has_already_learned(args.log, args.year, args.gp):
        print(f"Already learned from {args.gp} {args.year}, skipping.")
        exit(0)

    # Load prediction + features + actuals
    pred_df = pd.read_csv(args.pred_csv)
    feat_df = pd.read_csv(args.feature_csv)
    res_df = pd.read_csv(args.result_csv)[["Driver", "Position"]]

    # Merge to compute MAE before update
    merged = pd.merge(pred_df, res_df, on="Driver", how="inner")
    mae_before = mean_absolute_error(merged["Position"], merged["PredictedPosition"])
    print(f"MAE before update: {mae_before:.3f}")

    # Retrain
    if args.model_type == "xgb":
        model = load_xgb_model(args.model_path + ".json")
        model = retrain_xgb(model, feat_df, res_df, alpha=args.alpha)
        model.save_model(args.model_path + ".json")
        # after retrain, compute new MAE
        post_preds = model.predict(feat_df[feat_df['Driver'].isin(res_df['Driver'])][[
            'GridPosition','HistAvgLapTime','HistPitStops','HistStints',
            'DriverPoints','ConstructorPoints','Length','Corners','OvertakingDifficulty','DriverFormBoost'
        ]])
        mae_after = mean_absolute_error(res_df['Position'], post_preds)
    else:
        # neural net path
        # load model + scaler
        checkpoint = torch.load(args.model_path + ".pt")
        input_dim = len(['GridPosition','HistAvgLapTime','HistPitStops','HistStints',
                         'DriverPoints','ConstructorPoints','Length','Corners','OvertakingDifficulty','DriverFormBoost'])
        model, scaler = load_nn_model(args.model_path + ".pt", input_dim)
        model, scaler, mae_after = retrain_nn(model, scaler, feat_df, res_df, alpha=args.alpha, epochs=args.epochs)
        # save new state
        torch.save({'model_state': model.state_dict(), 'scaler': scaler}, args.model_path + ".pt")

    print(f"MAE after update: {mae_after:.3f}")

    # Log what we did
    log_update(args.log, args.year, args.gp, args.model_type, mae_before, mae_after)

