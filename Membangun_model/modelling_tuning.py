"""
================================================================================
MODELLING - ADVANCED (MANUAL LOGGING + HYPERPARAMETER TUNING + DAGSHUB)
================================================================================
File: modelling_tuning.py
Deskripsi: Training model dengan hyperparameter tuning, manual logging, dan DagsHub
Level: Advanced (4 pts)
Author: Muhammad Wildan
================================================================================
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*80)
print("MODELING - ADVANCED (MANUAL LOGGING + TUNING + DAGSHUB)")
print("="*80)

# ================================================================================
# DAGSHUB SETUP
# ================================================================================
print("\n[SETUP] Configuring DagsHub...")

# GANTI INI DENGAN CREDENTIALS DAGSHUB ANDA
import os

os.environ["DAGSHUB_USER_NAME"] = "Dannnnneeeee"
os.environ["DAGSHUB_TOKEN"] = "3c502b7943cd0698e71e141e4e5917cd4caa0ffb"

import dagshub
dagshub.init(
    repo_owner="Dannnnneeeee",
    repo_name="Eksperimen_SML_Muhammad-Wildan",
    mlflow=True
)

# Alternative: untuk lokal testing (comment bagian DagsHub di atas)
# mlflow.set_tracking_uri("file:./mlruns")

print(f"✓ MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# ================================================================================
# 1. LOAD DATA
# ================================================================================
print("\n[1/9] Loading preprocessed data...")
df = pd.read_csv('toyota_clean.csv')
print(f"✓ Data loaded: {len(df)} rows, {df.shape[1]} columns")

# ================================================================================
# 2. PREPARE FEATURES & TARGET
# ================================================================================
print("\n[2/9] Preparing features and target...")

feature_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize',
                   'model_encoded', 'transmission_encoded', 'fuelType_encoded']

missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"✗ Missing columns: {missing_cols}")
    exit(1)

X = df[feature_columns]
y = df['price']

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target range: ${y.min():,.0f} - ${y.max():,.0f}")

# ================================================================================
# 3. SPLIT DATA (Train, Validation, Test)
# ================================================================================
print("\n[3/9] Splitting data into Train/Val/Test...")

# Split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
)

print(f"✓ Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"✓ Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ================================================================================
# 4. SETUP MLFLOW EXPERIMENT
# ================================================================================
print("\n[4/9] Setting up MLflow experiment...")

experiment_name = "Toyota_Price_Prediction_Advanced"
mlflow.set_experiment(experiment_name)

print(f"✓ Experiment: {experiment_name}")
print("✓ Mode: Manual Logging (NO AUTOLOG)")

# ================================================================================
# 5. HYPERPARAMETER TUNING
# ================================================================================
print("\n[5/9] Hyperparameter tuning with GridSearchCV...")

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.05, 0.1, 0.15],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

print(f"✓ Testing {np.prod([len(v) for v in param_grid.values()])} combinations")

# Base model
base_model = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror'
)

# Grid Search
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

print("✓ Starting grid search... (this may take a while)")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"\n✓ Best parameters found:")
for param, value in best_params.items():
    print(f"   {param}: {value}")

# ================================================================================
# 6. TRAIN FINAL MODEL WITH BEST PARAMS
# ================================================================================
print("\n[6/9] Training final model with best parameters...")

# Start MLflow run
with mlflow.start_run(run_name=f"XGBoost_Tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    
    # MANUAL LOGGING START
    print("\n[MANUAL LOGGING] Logging parameters...")
    
    # Log all best parameters
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    
    # Log additional info
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("val_samples", len(X_val))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("total_features", X_train.shape[1])
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_param("cv_folds", 3)
    
    # Train final model
    final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, 
                                   objective='reg:squarederror')
    final_model.fit(X_train, y_train)
    
    print("✓ Model training completed")
    
    # ============================================================================
    # 7. EVALUATION ON ALL SETS
    # ============================================================================
    print("\n[7/9] Evaluating on Train/Val/Test sets...")
    
    def calculate_metrics(y_true, y_pred, set_name):
        """Calculate and return all metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Log to MLflow
        mlflow.log_metric(f"{set_name}_mse", mse)
        mlflow.log_metric(f"{set_name}_rmse", rmse)
        mlflow.log_metric(f"{set_name}_mae", mae)
        mlflow.log_metric(f"{set_name}_r2", r2)
        mlflow.log_metric(f"{set_name}_mape", mape)
        
        return {
            'mse': mse, 'rmse': rmse, 'mae': mae, 
            'r2': r2, 'mape': mape
        }
    
    # Predictions
    y_train_pred = final_model.predict(X_train)
    y_val_pred = final_model.predict(X_val)
    y_test_pred = final_model.predict(X_test)
    
    # Calculate metrics for all sets
    train_metrics = calculate_metrics(y_train, y_train_pred, "train")
    val_metrics = calculate_metrics(y_val, y_val_pred, "val")
    test_metrics = calculate_metrics(y_test, y_test_pred, "test")
    
    # Display results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Metric':<15} {'Train':<15} {'Validation':<15} {'Test':<15}")
    print("-"*80)
    print(f"{'RMSE ($)':<15} {train_metrics['rmse']:>14,.2f} {val_metrics['rmse']:>14,.2f} {test_metrics['rmse']:>14,.2f}")
    print(f"{'MAE ($)':<15} {train_metrics['mae']:>14,.2f} {val_metrics['mae']:>14,.2f} {test_metrics['mae']:>14,.2f}")
    print(f"{'R² Score':<15} {train_metrics['r2']:>14.4f} {val_metrics['r2']:>14.4f} {test_metrics['r2']:>14.4f}")
    print(f"{'MAPE (%)':<15} {train_metrics['mape']:>14.2f} {val_metrics['mape']:>14.2f} {test_metrics['mape']:>14.2f}")
    print("="*80)
    
    # Check for overfitting
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    if r2_diff < 0.05:
        print(" No overfitting detected (good generalization)")
    elif r2_diff < 0.1:
        print(" Slight overfitting (acceptable)")
    else:
        print(" Overfitting detected (consider regularization)")
    
    # ============================================================================
    # 8. CREATE CUSTOM ARTIFACTS (ADVANCED)
    # ============================================================================
    print("\n[8/9] Creating custom artifacts...")
    
    # ARTIFACT 1: Comprehensive Evaluation Plots (4 subplots)
    print("✓ Creating evaluation plots...")
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('XGBoost Model Evaluation - Advanced', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Predicted (Test Set)
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=30, label='Predictions')
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect')
    axes[0, 0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0, 0].set_title(f'Test Set: Actual vs Predicted (R²={test_metrics["r2"]:.4f})', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residual Plot
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, s=30)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals ($)', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Prediction Error ($)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title(f'Error Distribution (MAE=${test_metrics["mae"]:,.2f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    axes[1, 1].barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    axes[1, 1].set_xlabel('Importance Score', fontsize=12)
    axes[1, 1].set_ylabel('Features', fontsize=12)
    axes[1, 1].set_title('Feature Importance', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    eval_plot_path = 'evaluation_plots_advanced.png'
    plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(eval_plot_path)
    print(f"   ✓ Saved: {eval_plot_path}")
    plt.close()
    
    # ARTIFACT 2: Learning Curves (Train vs Val)
    print("✓ Creating learning curves...")
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    metrics_comparison = {
        'RMSE': [train_metrics['rmse'], val_metrics['rmse'], test_metrics['rmse']],
        'MAE': [train_metrics['mae'], val_metrics['mae'], test_metrics['mae']],
        'MAPE': [train_metrics['mape'], val_metrics['mape'], test_metrics['mape']]
    }
    
    x = np.arange(len(['Train', 'Validation', 'Test']))
    width = 0.25
    
    for i, (metric_name, values) in enumerate(metrics_comparison.items()):
        offset = width * i
        ax.bar(x + offset, values, width, label=metric_name, alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Error Value', fontsize=12)
    ax.set_title('Model Performance Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Train', 'Validation', 'Test'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    learning_curve_path = 'learning_curves.png'
    plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(learning_curve_path)
    print(f"   ✓ Saved: {learning_curve_path}")
    plt.close()
    
    # ARTIFACT 3: Prediction Error Analysis
    print("✓ Creating error analysis...")
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error by price range
    price_bins = pd.cut(y_test, bins=5)
    error_by_price = pd.DataFrame({
        'Price_Range': price_bins,
        'Absolute_Error': np.abs(y_test - y_test_pred)
    })
    error_summary = error_by_price.groupby('Price_Range')['Absolute_Error'].mean()
    
    axes[0].bar(range(len(error_summary)), error_summary.values, color='coral', alpha=0.7)
    axes[0].set_xlabel('Price Range', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error ($)', fontsize=12)
    axes[0].set_title('Error by Price Range', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels([f"${int(interval.left)}-{int(interval.right)}" 
                             for interval in error_summary.index], rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Percentage error distribution
    pct_errors = np.abs((y_test - y_test_pred) / y_test) * 100
    axes[1].hist(pct_errors, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1].axvline(x=pct_errors.median(), color='r', linestyle='--', lw=2, 
                    label=f'Median: {pct_errors.median():.2f}%')
    axes[1].set_xlabel('Percentage Error (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Percentage Error Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_analysis_path = 'error_analysis.png'
    plt.savefig(error_analysis_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(error_analysis_path)
    print(f"   ✓ Saved: {error_analysis_path}")
    plt.close()
    
    # ARTIFACT 4: Model Comparison Summary (JSON)
    print("✓ Creating model summary...")
    model_summary = {
        'model_name': 'XGBoost Regressor (Tuned)',
        'best_parameters': best_params,
        'training_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'features': feature_columns
        },
        'performance': {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        },
        'overfitting_check': {
            'r2_difference': float(r2_diff),
            'status': 'Good' if r2_diff < 0.05 else 'Acceptable' if r2_diff < 0.1 else 'Overfitting'
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_path = 'model_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(model_summary, f, indent=4)
    mlflow.log_artifact(summary_path)
    print(f"   ✓ Saved: {summary_path}")
    
    # ARTIFACT 5: Feature Importance CSV
    print("✓ Creating feature importance file...")
    importance_df.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    print(f"   ✓ Saved: feature_importance.csv")
    
    # Log the model itself
    print("✓ Logging model to MLflow...")
    mlflow.xgboost.log_model(final_model, "model")
    
    # Get run ID
    run_id = mlflow.active_run().info.run_id
    print(f"\n✓ MLflow Run ID: {run_id}")

# ================================================================================
# 9. FINAL SUMMARY
# ================================================================================
print("\n" + "="*80)
print("✓ MODELING ADVANCED SELESAI!")
print("="*80)

print("\n FINAL TEST RESULTS:")
print(f"   • R² Score: {test_metrics['r2']:.4f} ({test_metrics['r2']*100:.2f}% variance explained)")
print(f"   • RMSE: ${test_metrics['rmse']:,.2f}")
print(f"   • MAE: ${test_metrics['mae']:,.2f}")
print(f"   • MAPE: {test_metrics['mape']:.2f}%")

print("\n ARTIFACTS GENERATED:")
print("   ✓ evaluation_plots_advanced.png")
print("   ✓ learning_curves.png")
print("   ✓ error_analysis.png")
print("   ✓ model_summary.json")
print("   ✓ feature_importance.csv")
print("   ✓ Trained model")

print("\n ADVANCED FEATURES IMPLEMENTED:")
print("   ✓ Hyperparameter Tuning (GridSearchCV)")
print("   ✓ Manual Logging (NO autolog)")
print("   ✓ DagsHub Integration")
print("   ✓ 5 Custom Artifacts (autolog + 5 additional)")
print("   ✓ Train/Val/Test Split")
print("   ✓ Overfitting Check")

print("\n VIEW RESULTS:")
print(f"   DagsHub: https://dagshub.com/Dannnnneeeee/Eksperimen_SML_Muhammad-Wildan")
print(f"   Or MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")

print("\n SCREENSHOTS NEEDED:")
print("   1. screenshoot_dashboard.jpg - DagsHub experiments page")
print("   2. screenshoot_artifak.jpg - Artifacts tab showing all 6+ items")

print("\n" + "="*80)
print("✨ READY FOR ADVANCED (4 PTS) SUBMISSION!")
print("="*80)