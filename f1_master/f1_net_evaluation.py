import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# ----------------------------------------------------
# Helper: Evaluate a single race
# ----------------------------------------------------
def evaluate_race(pred_path, result_path):
    """Evaluates a single race given prediction and result CSVs."""
    preds = pd.read_csv(pred_path)
    actual = pd.read_csv(result_path)

    # Auto-detect driver column
    pred_driver_col = next((c for c in preds.columns if "driver" in c.lower()), None)
    actual_driver_col = next((c for c in actual.columns if "driver" in c.lower()), None)
    finish_col = next(
        (col for col in ["FinishPosition", "FinishPosition_actual", "Position", "Position_actual"]
         if col in actual.columns),
        None
    )

    if not (pred_driver_col and actual_driver_col and finish_col):
        print(f"?? Skipping {os.path.basename(pred_path)} - missing columns.")
        return None

    preds[pred_driver_col] = preds[pred_driver_col].astype(str).str.strip().str.upper()
    actual[actual_driver_col] = actual[actual_driver_col].astype(str).str.strip().str.upper()

    merged = preds.merge(actual, left_on=pred_driver_col, right_on=actual_driver_col, suffixes=("_pred", "_actual"))
    if merged.empty:
        print(f"?? Skipping {os.path.basename(pred_path)} - no driver overlap.")
        return None

    # Evaluate metrics
    mae = mean_absolute_error(merged[finish_col], merged["PredictedPosition"])
    acc = (1 - mae / 20) * 100

    top3 = np.mean(
        merged.nsmallest(3, "PredictedPosition")[pred_driver_col].isin(
            merged.nsmallest(3, finish_col)[actual_driver_col]
        )
    ) * 100
    top10 = np.mean(
        merged.nsmallest(10, "PredictedPosition")[pred_driver_col].isin(
            merged.nsmallest(10, finish_col)[actual_driver_col]
        )
    ) * 100
    winner = 100.0 if merged.nsmallest(1, "PredictedPosition")[pred_driver_col].iloc[0] == \
                     merged.nsmallest(1, finish_col)[actual_driver_col].iloc[0] else 0.0

    gp_name = os.path.basename(pred_path).replace(".csv", "").split("_predicted_")[-1].replace("_", " ")

    return {
        "GrandPrix": gp_name,
        "MAE": mae,
        "Accuracy(%)": acc,
        "Top3(%)": top3,
        "Top10(%)": top10,
        "Winner(%)": winner
    }

# ----------------------------------------------------
# Main function
# ----------------------------------------------------
def main():
    pred_dir = os.path.join("predictions", "csv")
    result_dir = os.path.join("actual_results")
    eval_dir = os.path.join("evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    results = []
    for file in os.listdir(pred_dir):
        if not file.endswith(".csv"):
            continue
        year_gp = file.replace("_predicted_", "_")
        result_file = year_gp.replace(".csv", "_results.csv")
        result_path = os.path.join(result_dir, result_file)
        pred_path = os.path.join(pred_dir, file)

        if os.path.exists(result_path):
            stats = evaluate_race(pred_path, result_path)
            if stats:
                results.append(stats)
        else:
            print(f"? Missing result file for {file}")

    if not results:
        print("?? No valid races evaluated.")
        return

    df = pd.DataFrame(results)
    summary = pd.DataFrame({
        "Metric": ["MAE", "Accuracy(%)", "Top3(%)", "Top10(%)", "Winner(%)"],
        "Average": [
            df["MAE"].mean(),
            df["Accuracy(%)"].mean(),
            df["Top3(%)"].mean(),
            df["Top10(%)"].mean(),
            df["Winner(%)"].mean(),
        ],
    })

    print("\n?? SEASON-WIDE EVALUATION SUMMARY\n")
    print(df.to_string(index=False))
    print("\n--------------------------------------")
    print(summary.to_string(index=False))

    df.to_csv(os.path.join(eval_dir, "season_stats.csv"), index=False)
    summary.to_csv(os.path.join(eval_dir, "season_summary.csv"), index=False)
    print("\n? Evaluation reports saved under /evaluation/")

if __name__ == "__main__":
    main()

