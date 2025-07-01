import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

df = pd.read_csv('../Dataset/Student Dataset(features=20).csv')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

features = [
    'total_submissions', 'problem_tried', 'problem_solved',
    'avg_completion_rate_delta', 'avg_inverse_solving_rate',
    'avg_submissions_per_day', 'activity_span',
    'percentage_submissions_night_hours', 'percentage_submissions_study_hours',
    'submission_hour_variance', 'submission_efficiency', 'avg_attempts_per_problem',
    'active_days', 'active_ratio', 'avg_assignment_score',
    'assignment_score_variance', 'weekend_submission_ratio', 'focus_index',
    'problem_solved_rate', 'assignment_completion_rate'
]
target = ['qt', 'th', 'ck']

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

base_model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    learning_rate=0.05,
    n_estimators=100,
    max_depth=7,
    objective='regression',
    metric='rmse',
    random_state=42
)

model = MultiOutputRegressor(base_model)
model.fit(X_train_pca, y_train_scaled)

y_pred_scaled = model.predict(X_test_pca)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")

input_df = pd.read_csv('../Input - Output/input(features=20).csv')
X_input = input_df[features]

if X_input.isnull().values.any():
    X_input = X_input.fillna(X_input.mean())

X_input_scaled = scaler_X.transform(X_input)
X_input_pca = pca.transform(X_input_scaled)

y_input_scaled = model.predict(X_input_pca)
y_input_pred = scaler_y.inverse_transform(y_input_scaled)

input_df['predicted_qt'] = y_input_pred[:, 0]
input_df['predicted_th'] = y_input_pred[:, 1]
input_df['predicted_ck'] = y_input_pred[:, 2]

input_df.to_csv('output3.csv', index=False)
print("✅ Predictions saved to output3.csv")
