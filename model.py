# This is a model to predict the race winners/result.
#
# @author : mjNotFound-19
# @modified : 5/29/25

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

def gradient_f1(df: pd.DataFrame, feature_cols: list, label_col: str):
	""" 
    	Train an XGBoost model to predict finishing position.

    	:param df: Input DataFrame with features and target
    	:param feature_cols: List of column names used as features
    	:param label_col: Name of the target column (finishing position)
    	:return: trained model
    	"""
	X = df[feature_cols]
	Y = df[label_col] - 1

	X_train, X_test, Y_train, Y_test = test_train_split(X,Y, test_size = 0.2, random_state = 42, stratify = Y)

	model = xgb.XGBClassifier(objective = "multi:softprob", num_class = 20, max_depth = 6, learning_rate = 0.1, n_estimators = 10, sub_sample = 0.8, colsample_bytree = 0.8, use_label_encoder = False, eval_metric = 'mlogloss')

	model.fit(X_train, Y_train)

	Y_pred = model.predict(X_test)
	acc = accuracy_score(Y_test, Y_pred)
	mae = mean_absolute_error(Y_test + 1, Y_pred + 1)
	print(f"Accuracy: {acc:.3f}, MAE: {mae:.2f}")

	joblib.dump(model, "gradient_f1.pkl")
	return model
