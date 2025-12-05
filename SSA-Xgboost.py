import os
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy.io import savemat
import scipy.io as sio
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from mealpy import FloatVar, SSA, WOA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import time
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from joblib import Parallel, delayed
import warnings
import joblib

overall_start_time = time.time()

df = pd.read_csv('real_data.csv')
df = df.iloc[:, :]
X = df.drop(['y'], axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
data_train_x = X_train
data_train_y = y_train
data_test_x = X_test
data_test_y = y_test

data_train_x_np = data_train_x.values
data_train_y_np = data_train_y.values
data_test_x_np = data_test_x.values
data_test_y_np = data_test_y.values

X_mean, y_mean = data_train_x_np.mean(0), data_train_y_np.mean(0)
X_std, y_std = data_train_x_np.std(0), data_train_y_np.std(0)

data_train_x_nor = (data_train_x_np - X_mean) / X_std
data_test_x_nor = (data_test_x_np - X_mean) / X_std

data_train_y_nor = (data_train_y_np - y_mean) / y_std
data_test_y_nor = (data_test_y_np - y_mean) / y_std

scaler_params = {
    'X_mean': X_mean,
    'X_std': X_std,
    'y_mean': y_mean,
    'y_std': y_std
}
np.save('scaler_parameters.npy', scaler_params)

data_train_y_nor = pd.Series(data_train_y_nor).reset_index(drop=True)

def evaluate_regress(y_pre, y_true):
    y_pre = np.asarray(y_pre).flatten()
    y_true = np.asarray(y_true).flatten()

    abs_errors = np.abs(y_pre - y_true)
    MAE = np.mean(abs_errors)

    # Avoid division by zero in MAPE
    safe_denominator = np.where(y_true == 0, 1e-8, y_true)
    MAPE = np.mean(np.abs((y_pre - y_true) / safe_denominator))

    squared_errors = (y_pre - y_true) ** 2
    MSE = np.mean(squared_errors)
    RMSE = np.sqrt(MSE)

    R2 = r2_score(y_true, y_pre)

    print(f'MAE为: {MAE:.6f}')
    print(f'MAPE为: {MAPE:.6f}')
    print(f'MSE为: {MSE:.6f}')
    print(f'RMSE为: {RMSE:.6f}')
    print(f'R2为: {R2:.6f}')

    return MAE, MAPE, MSE, RMSE, R2


class XGBoostRegressorWrapper:
    def __init__(self,
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.1,
                 min_child_weight=1,
                 gamma=0,
                 subsample=1,
                 colsample_bytree=1,
                 alpha=0,
                 lambd=1,
                 random_state=42,
                 n_jobs=-1
                 ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=alpha,
            reg_lambda=lambd,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# Precompute KFold indices to avoid repeated computation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_indices = list(kf.split(data_train_x_np))

best_fitness_history = []
best_params_history = []


# Function to train and evaluate a single fold
def train_and_evaluate_fold(params, train_idx, val_idx):
    try:
        # Extract fold data using precomputed indices
        X_train_fold = data_train_x_np[train_idx]
        X_val_fold = data_train_x_np[val_idx]
        y_train_fold = data_train_y_np[train_idx]
        y_val_fold = data_train_y_np[val_idx]

        # Standardization using precomputed global parameters
        X_train_fold_scaled = (X_train_fold - X_mean) / X_std
        X_val_fold_scaled = (X_val_fold - X_mean) / X_std
        y_train_fold_scaled = (y_train_fold - y_mean) / y_std
        y_val_fold_scaled = (y_val_fold - y_mean) / y_std

        # Create and train model
        model = XGBoostRegressorWrapper(**params, random_state=42, n_jobs=1)
        model.fit(X_train_fold_scaled, y_train_fold_scaled)
        y_pred = model.predict(X_val_fold_scaled)

        return mean_absolute_error(y_val_fold_scaled, y_pred)
    except Exception as e:
        print(f"Error in fold training: {e}")
        return float('inf')


def evaluate_model(solution):
    # Convert solution to model parameters
    params = {
        'max_depth': max(min(int(solution[0]), 8), 3),
        'learning_rate': max(min(solution[1], 0.2), 0.01),
        'n_estimators': max(min(int(solution[2]), 300), 50),
        'min_child_weight': max(min(int(solution[3]), 10), 1),
        'gamma': max(min(solution[4], 0.5), 0),
        'subsample': max(min(solution[5], 1.0), 0.6),
        'colsample_bytree': max(min(solution[6], 1.0), 0.6),
        'alpha': max(min(solution[7], 1.0), 0.001),
        'lambd': max(min(solution[8], 1.0), 0.1)
    }

    # Parallel execution of cross-validation folds
    mae_scores = Parallel(n_jobs=min(5, os.cpu_count()))(
        delayed(train_and_evaluate_fold)(params, train_idx, val_idx)
        for train_idx, val_idx in cv_indices
    )

    # Filter out infinite values (errors)
    valid_scores = [score for score in mae_scores if score != float('inf')]

    if valid_scores:
        mean_mae = np.mean(valid_scores)

        # Update best fitness and corresponding hyperparameters
        if len(best_fitness_history) < 150 or mean_mae < best_fitness_history[-1]:
            best_fitness_history.append(mean_mae)
            best_params_history.append(solution)

        if len(best_fitness_history) > 150:
            best_fitness_history.pop(0)
            best_params_history.pop(0)

        return mean_mae
    else:
        return float('inf')


# Define parameter ranges
param_grid = {
    "obj_func": evaluate_model,
    "bounds": [
        FloatVar(lb=3, ub=8),  # max_depth (扩展到3-8)
        FloatVar(lb=0.01, ub=0.2),  # learning_rate (扩展到0.01-0.2)
        FloatVar(lb=50, ub=300),  # n_estimators (保持50-300)
        FloatVar(lb=1, ub=10),  # min_child_weight (扩展到1-10)
        FloatVar(lb=0, ub=0.5),  # gamma (扩展到0-0.5)
        FloatVar(lb=0.6, ub=1.0),  # subsample (扩展到0.6-1.0)
        FloatVar(lb=0.6, ub=1.0),  # colsample_bytree (扩展到0.6-1.0)
        FloatVar(lb=0.001, ub=1.0),  # reg_alpha (扩展到0.001-1.0)
        FloatVar(lb=0.1, ub=1.0)  # reg_lambda (保持0.1-1.0)
    ],
    "minmax": "min"
}

# Set parameters for the SSA algorithm
epoch = 50
pop_size = 30
SSA_model = SSA.OriginalSSA(epoch=epoch, pop_size=pop_size)

print("Starting SSA optimization...")
# Solve the optimization problem
SSA_best = SSA_model.solve(param_grid)
print("SSA optimization completed.")

# Create a DataFrame to save the best fitness and hyperparameters
best_history_df = pd.DataFrame({
    'Best_Fitness': best_fitness_history,
    'Max_Depth': [int(param[0]) for param in best_params_history],
    'Learning_Rate': [param[1] for param in best_params_history],
    'N_Estimators': [int(param[2]) for param in best_params_history],
    'Min_Child_Weight': [int(param[3]) for param in best_params_history],
    'Gamma': [param[4] for param in best_params_history],
    'Subsample': [param[5] for param in best_params_history],
    'Colsample_ByTree': [param[6] for param in best_params_history],
    'Alpha': [param[7] for param in best_params_history],
    'Lambda': [param[8] for param in best_params_history]
})

# Best parameters
final_best_params = SSA_best.solution

# Save DataFrame to Excel
output_path = 'optimized_paras_SSA_XGBoost.xlsx'
with pd.ExcelWriter(output_path) as writer:
    best_history_df.to_excel(writer, sheet_name='Best_Parameters', index=False)
    pd.DataFrame([final_best_params], columns=[f'Final_Param_{i + 1}' for i in range(len(final_best_params))]).to_excel(
        writer, sheet_name='Final_Best_Parameters', index=False)

# Retrieve the best parameters
best_max_depth = int(final_best_params[0])
best_learning_rate = final_best_params[1]
best_n_estimators = int(final_best_params[2])
best_min_child_weight = int(final_best_params[3])
best_gamma = final_best_params[4]
best_subsample = final_best_params[5]
best_colsample_bytree = final_best_params[6]
best_alpha = final_best_params[7]
best_lambda = final_best_params[8]

print("Training final model with best parameters...")

best_model = XGBoostRegressorWrapper(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    learning_rate=best_learning_rate,
    min_child_weight=best_min_child_weight,
    gamma=best_gamma,
    subsample=best_subsample,
    colsample_bytree=best_colsample_bytree,
    alpha=best_alpha,
    lambd=best_lambda,
    random_state=42,
    n_jobs=-1
)
best_model.fit(data_train_x_nor, data_train_y_nor)


joblib.dump(best_model, 'SSA-Xgboost.pkl')

# Prediction
y_pred_test_nor = best_model.predict(data_test_x_nor)
y_pred_train_nor = best_model.predict(data_train_x_nor)

# Anti-standardization
y_pred_test = y_pred_test_nor * y_std + y_mean
y_pred_test1 = y_pred_test.reshape(len(y_pred_test), 1)
data_test_y1 = data_test_y_np.reshape(len(data_test_y_np), 1)

y_pred_train = y_pred_train_nor * y_std + y_mean
y_pred_train1 = y_pred_train.reshape(len(y_pred_train), 1)
data_train_y1 = data_train_y_np.reshape(len(data_train_y_np), 1)

print("Calculating evaluation metrics...")
# Calculation error
T_MAE, T_MAPE, T_MSE, T_RMSE, T_R2 = evaluate_regress(y_pred_test1, data_test_y1)
R_MAE, R_MAPE, R_MSE, R_RMSE, R_R2 = evaluate_regress(y_pred_train1, data_train_y1)

# Create a DataFrame
errors_test = pd.DataFrame({
    'test—Metric': ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2'],
    'test—Value': [T_MAE, T_MAPE, T_MSE, T_RMSE, T_R2]
})
errors_train = pd.DataFrame({
    'train—Metric': ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2'],
    'train—Value': [R_MAE, R_MAPE, R_MSE, R_RMSE, R_R2]
})

# Reconstruct predictions and true values
predictions = np.concatenate((y_pred_train[:, np.newaxis], y_pred_test[:, np.newaxis]), axis=0)
truevalues = np.concatenate((data_train_y_np[:, np.newaxis], data_test_y_np[:, np.newaxis]), axis=0)
predictions = predictions.ravel()
truevalues = truevalues.ravel()

results_df = pd.DataFrame({'Predictions': predictions, 'True Values': truevalues})

# Save the results to Excel
output_path_results = 'results_optimized_SSA_XGBoost.xlsx'
with pd.ExcelWriter(output_path_results) as writer:
    errors_test.to_excel(writer, sheet_name='Test_Errors', index=False)
    errors_train.to_excel(writer, sheet_name='Train_Errors', index=False)
    results_df.to_excel(writer, sheet_name='Predictions', index=False)

# Record the end time of the whole process.
overall_end_time = time.time()
overall_total_time = overall_end_time - overall_start_time
print(f"Total time taken: {overall_total_time:.2f} seconds")

# Save total runtime to an Excel file
with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
    process_time_df = pd.DataFrame({
        'Total_Process_Time_Seconds': [overall_total_time]
    })
    process_time_df.to_excel(writer, sheet_name='Process_Time', index=False)

print(f'Save to: {output_path} excel ')