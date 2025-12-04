"""
Train Best Model Script
=======================
Trains an XGBoost model with class imbalance handling and threshold optimization.
Saves the model, metadata, and feature importance for the production API.
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from xgboost import XGBClassifier

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.ml.preprocessing import NoShowPreprocessor
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = project_root / 'config' / 'ml_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_best_model():
    logger.info("Starting model training pipeline...")
    
    # 1. Load Config and Data
    config = load_config()
    data_path = project_root / 'data' / 'processed' / 'appointments_features.csv'
    
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape}")
    
    # 2. Preprocessing
    preprocessor = NoShowPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = preprocessor.feature_names_
    logger.info(f"Preprocessing complete. Features: {len(feature_names)}")
    
    # 3. Train XGBoost
    # Optimized hyperparameters from Week 6 Tuning
    logger.info("Training XGBoost with optimized hyperparameters...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=7,
        gamma=0.2,
        reg_alpha=0.01,
        reg_lambda=0.01,
        scale_pos_weight=1.0,  # Tuned value
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_processed, y_train)
    logger.info("Model training complete.")
    
    # 4. Threshold Optimization
    # We want high recall to catch no-shows, but reasonable precision
    y_probs = model.predict_proba(X_test_processed)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Find threshold that gives at least 0.75 Recall (or best F1 if not possible)
    target_recall = 0.75
    optimal_threshold = 0.5
    best_f1 = 0
    
    for p, r, t in zip(precisions, recalls, thresholds):
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        if r >= target_recall:
            # Among those with good recall, maximize F1/Precision
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = t
                
    logger.info(f"Optimal Threshold selected: {optimal_threshold:.4f} (Target Recall: {target_recall})")
    
    # 5. Evaluation
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_probs)),
        "threshold": float(optimal_threshold)
    }
    
    logger.info(f"Test Metrics: {metrics}")
    
    # 6. Feature Importance & SHAP
    importance = model.feature_importances_
    # Map back to feature names
    feat_imp = []
    for name, imp in zip(feature_names, importance):
        feat_imp.append({"feature": name, "importance": float(imp)})
        
    # Sort by importance
    feat_imp.sort(key=lambda x: x['importance'], reverse=True)
    top_features = feat_imp[:20] # Keep top 20
    
    # SHAP Explainer
    import shap
    logger.info("Generating SHAP explainer...")
    # Use TreeExplainer for XGBoost
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        logger.warning(f"SHAP TreeExplainer failed with model: {e}. Trying with booster...")
        try:
            explainer = shap.TreeExplainer(model.get_booster())
        except Exception as e2:
             logger.error(f"SHAP TreeExplainer failed with booster: {e2}. Skipping SHAP.")
             explainer = None
    
    # 7. Save Artifacts
    output_dir = project_root / 'models' / 'production'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Model
    joblib.dump(model, output_dir / 'model.joblib')
    
    # Save Preprocessor (IMPORTANT for API)
    preprocessor_save = {
        'preprocessor': preprocessor.preprocessor_,
        'feature_names': feature_names,
        'valid_numeric': preprocessor._valid_numeric,
        'valid_categorical': preprocessor._valid_categorical,
        'valid_binary': preprocessor._valid_binary
    }
    joblib.dump(preprocessor_save, output_dir / 'preprocessor.joblib')
    
    # Save SHAP Explainer
    joblib.dump(explainer, output_dir / 'shap_explainer.joblib')
    
    # Save Metadata
    metadata = {
        "model_name": "xgboost_optimized_v2",
        "model_version": "2.1.0",
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "optimal_threshold": float(optimal_threshold),
        "parameters": model.get_params()
    }
    
    with open(output_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Save Feature Importance
    with open(output_dir / 'feature_importance.json', 'w') as f:
        json.dump(top_features, f, indent=4)
        
    logger.info(f"Artifacts saved to {output_dir}")

if __name__ == "__main__":
    train_best_model()
