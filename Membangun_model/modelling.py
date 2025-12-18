"""
================================================================================
MODELLING - BASIC (AUTOLOG)
================================================================================
File: modelling.py
Deskripsi: Training model XGBoost dengan MLflow Autolog + Evaluation
Level: Basic (2 pts)
Author: Muhammad Wildan
================================================================================
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODELING - BASIC (MLFLOW AUTOLOG - XGBOOST)")
print("="*80)

# ================================================================================
# 1. LOAD DATA
# ================================================================================
print("\n[1/7] Loading preprocessed data...")
df = pd.read_csv('toyota_preprocessing.csv')
print(f"✓ Data loaded: {len(df)} rows, {df.shape[1]} columns")

# ================================================================================
# 2. PREPARE FEATURES & TARGET
# ================================================================================
print("\n[2/7] Preparing features and target...")

# Gunakan kolom yang sudah di-encode
feature_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize',
                   'model_encoded', 'transmission_encoded', 'fuelType_encoded']

# Check if columns exist
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"✗ Missing columns: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

X = df[feature_columns]
y = df['price']

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Target range: ${y.min():,.0f} - ${y.max():,.0f}")

# ================================================================================
# 3. SPLIT DATA
# ================================================================================
print("\n[3/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ================================================================================
# 4. SETUP MLFLOW
# ================================================================================
print("\n[4/7] Setting up MLflow...")

# Set experiment name
mlflow.set_experiment("Toyota_Price_Prediction_Basic")

# Enable autolog untuk XGBoost
mlflow.xgboost.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

print("✓ MLflow XGBoost autolog enabled")
print("✓ Experiment: Toyota_Price_Prediction_Basic")

# ================================================================================
# 5. TRAIN MODEL
# ================================================================================
print("\n[5/7] Training XGBoost model...")

with mlflow.start_run(run_name="XGBoost_Basic"):
    # Hyperparameters untuk XGBoost
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }
    
    print("✓ Model: XGBoost Regressor")
    print("✓ Training in progress...")
    
    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    print("✓ Training completed!")
    
    # ============================================================================
    # 6. EVALUATION ON TEST SET
    # ============================================================================
    print("\n[6/7] Evaluating model on test set...")
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Calculate accuracy percentage (based on R² score)
    accuracy = r2 * 100
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"Mean Absolute Error (MAE):      ${mae:,.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R² Score:                       {r2:.4f}")
    print(f"Model Accuracy:                 {accuracy:.2f}%")
    print("="*80)
    
    # Interpretation
    print("\n📊 INTERPRETATION:")
    if r2 >= 0.9:
        print("✓ Excellent! Model explains >90% of price variance")
    elif r2 >= 0.8:
        print("✓ Very Good! Model explains >80% of price variance")
    elif r2 >= 0.7:
        print("✓ Good! Model explains >70% of price variance")
    else:
        print("⚠ Fair. Model could be improved with feature engineering or tuning")
    
    print(f"✓ On average, predictions are off by ${mae:,.2f}")
    print(f"✓ Prediction error is approximately {mape:.1f}% of actual price")
    
    # Sample predictions comparison
    print("\n📋 SAMPLE PREDICTIONS (First 10 test samples):")
    print("-" * 80)
    print(f"{'Actual Price':<20} {'Predicted Price':<20} {'Difference':<20} {'Error %'}")
    print("-" * 80)
    
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        diff = predicted - actual
        error_pct = abs(diff / actual) * 100
        
        print(f"${actual:>18,.2f} ${predicted:>18,.2f} ${diff:>18,.2f} {error_pct:>8.2f}%")
    
    print("-" * 80)
    
    # Create visualization
    print("\n📈 Creating evaluation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('XGBoost Model Evaluation', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=30)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0, 0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals ($)', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Prediction Error ($)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    axes[1, 1].barh(importance_df['Feature'], importance_df['Importance'])
    axes[1, 1].set_xlabel('Importance Score', fontsize=12)
    axes[1, 1].set_ylabel('Features', fontsize=12)
    axes[1, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = 'model_evaluation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Evaluation plot saved: {plot_path}")
    
    plt.close()
    
    print("✓ Model and artifacts logged to MLflow")

# ================================================================================
# 7. FINAL SUMMARY
# ================================================================================
print("\n" + "="*80)
print("✓ MODELING BASIC SELESAI!")
print("="*80)
print("\n EXPERIMENT INFO:")
print(f"   Name: Toyota_Price_Prediction_Basic")
print(f"   Algorithm: XGBoost Regressor")
print(f"   Accuracy: {accuracy:.2f}%")
print(f"   RMSE: ${rmse:,.2f}")
print("="*80)
print("\n✨ Ready untuk screenshot dan submission!")
print("="*80)