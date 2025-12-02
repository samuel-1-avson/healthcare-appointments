"""
Model Evaluation Module
=======================
Comprehensive model evaluation for no-show prediction.

This module provides:
- Standard classification metrics
- Business-relevant metrics (cost-based)
- Visualization of results
- Threshold optimization
- Statistical tests for model comparison
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import json

# sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    log_loss
)
from sklearn.calibration import calibration_curve

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    brier_score: float
    log_loss: float
    confusion_matrix: np.ndarray
    threshold: float = 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc,
            'average_precision': self.average_precision,
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'threshold': self.threshold
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation for classification.
    
    This class provides:
    - Standard metrics computation
    - Business cost analysis
    - Visualization generation
    - Threshold optimization
    - Model comparison
    
    Attributes
    ----------
    config : dict
        ML configuration dictionary
    results : dict
        Dictionary of model_name -> EvaluationResult
    
    Example
    -------
    >>> evaluator = ModelEvaluator(ml_config)
    >>> results = evaluator.evaluate_all(models, X_test, y_test)
    >>> evaluator.plot_roc_curves()
    >>> evaluator.plot_confusion_matrices()
    """
    
    def __init__(self, config: dict):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        config : dict
            ML configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.ml.ModelEvaluator")
        
        # Store results
        self.results: Dict[str, EvaluationResult] = {}
        self.roc_curves: Dict[str, Tuple] = {}
        self.pr_curves: Dict[str, Tuple] = {}
        
        # Business config
        self.business_config = config['evaluation'].get('business', {})
        
        # Plotting config
        self.figures_dir = Path(config['output'].get('figures_dir', 'outputs/figures/ml'))
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("ModelEvaluator initialized")
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        threshold: float = 0.5
    ) -> EvaluationResult:
        """
        Evaluate a single model on test data.
        
        Parameters
        ----------
        model : estimator
            Trained sklearn-compatible model
        X_test : np.ndarray
            Test feature matrix
        y_test : np.ndarray
            Test target vector
        model_name : str
            Name of the model
        threshold : float
            Classification threshold
        
        Returns
        -------
        EvaluationResult
            Evaluation results container
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Compute metrics
        result = EvaluationResult(
            model_name=model_name,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, y_pred_proba),
            average_precision=average_precision_score(y_test, y_pred_proba),
            brier_score=brier_score_loss(y_test, y_pred_proba),
            log_loss=log_loss(y_test, y_pred_proba),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            threshold=threshold
        )
        
        # Store ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        self.roc_curves[model_name] = (fpr, tpr, thresholds)
        
        # Store PR curve data
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        self.pr_curves[model_name] = (precision, recall, thresholds)
        
        # Store result
        self.results[model_name] = result
        
        self._log_results(result)
        
        return result
    
    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate all models on test data.
        
        Parameters
        ----------
        models : dict
            Dictionary of model_name -> trained model
        X_test : np.ndarray
            Test feature matrix
        y_test : np.ndarray
            Test target vector
        threshold : float
            Classification threshold
        
        Returns
        -------
        dict
            Dictionary of model_name -> EvaluationResult
        """
        self.logger.info("="*60)
        self.logger.info(f"Evaluating {len(models)} models on test set...")
        self.logger.info(f"Test set size: {len(y_test):,}")
        self.logger.info(f"Positive class rate: {y_test.mean():.1%}")
        self.logger.info("="*60)
        
        for name, model in models.items():
            self.evaluate_model(model, X_test, y_test, name, threshold)
        
        return self.results
    
    def _log_results(self, result: EvaluationResult) -> None:
        """Log evaluation results."""
        cm = result.confusion_matrix
        tn, fp, fn, tp = cm.ravel()
        
        self.logger.info(f"\n{result.model_name} Results:")
        self.logger.info(f"  Accuracy:   {result.accuracy:.4f}")
        self.logger.info(f"  Precision:  {result.precision:.4f}")
        self.logger.info(f"  Recall:     {result.recall:.4f}")
        self.logger.info(f"  F1 Score:   {result.f1:.4f}")
        self.logger.info(f"  ROC-AUC:    {result.roc_auc:.4f}")
        self.logger.info(f"  Avg Prec:   {result.average_precision:.4f}")
        self.logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table of all evaluated models.
        
        Returns
        -------
        pd.DataFrame
            Comparison table sorted by primary metric
        """
        if not self.results:
            raise ValueError("No models evaluated yet.")
        
        rows = []
        for name, result in self.results.items():
            rows.append({
                'Model': name,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1 Score': result.f1,
                'ROC-AUC': result.roc_auc,
                'Avg Precision': result.average_precision,
                'Brier Score': result.brier_score
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by primary metric
        primary_metric = self.config['evaluation']['primary_metric']
        metric_col_map = {
            'roc_auc': 'ROC-AUC',
            'f1': 'F1 Score',
            'precision': 'Precision',
            'recall': 'Recall',
            'accuracy': 'Accuracy'
        }
        sort_col = metric_col_map.get(primary_metric, 'ROC-AUC')
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        
        return df
    
    def optimize_threshold(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.
        
        Parameters
        ----------
        model : estimator
            Trained model
        X_val : np.ndarray
            Validation feature matrix
        y_val : np.ndarray
            Validation target vector
        model_name : str
            Name of model
        metric : str
            Metric to optimize ('f1', 'precision', 'recall')
        
        Returns
        -------
        tuple
            (optimal_threshold, best_score)
        """
        self.logger.info(f"Optimizing threshold for {model_name} (metric: {metric})...")
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred, zero_division=0)
            else:
                score = f1_score(y_val, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        self.logger.info(f"Optimal threshold: {best_threshold:.2f} ({metric}: {best_score:.4f})")
        
        return best_threshold, best_score
    
    def calculate_business_impact(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Calculate business impact of predictions.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        model_name : str
            Name of model
        
        Returns
        -------
        dict
            Business impact metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Get costs from config
        cost_fn = self.business_config.get('cost_per_false_negative', 150)  # Missed no-show
        cost_fp = self.business_config.get('cost_per_false_positive', 10)   # Unnecessary intervention
        cost_noshow = self.business_config.get('cost_per_noshow', 150)
        
        # Calculate costs
        baseline_cost = (fn + tp) * cost_noshow  # Cost if we did nothing
        model_cost = fn * cost_fn + fp * cost_fp  # Cost with model
        savings = baseline_cost - model_cost
        
        # Prevented no-shows (true positives we can intervene on)
        # Assume 30% of predicted no-shows are prevented by intervention
        intervention_success_rate = 0.30
        prevented_noshows = int(tp * intervention_success_rate)
        prevented_savings = prevented_noshows * cost_noshow
        
        impact = {
            'model_name': model_name,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'baseline_cost': baseline_cost,
            'model_cost': model_cost,
            'gross_savings': savings,
            'intervention_cost': fp * cost_fp,
            'missed_noshow_cost': fn * cost_fn,
            'prevented_noshows': prevented_noshows,
            'prevented_savings': prevented_savings,
            'net_savings': prevented_savings - (fp * cost_fp)
        }
        
        self.logger.info(f"\nBusiness Impact for {model_name}:")
        self.logger.info(f"  Baseline cost (no model): ${baseline_cost:,.0f}")
        self.logger.info(f"  Model-related cost: ${model_cost:,.0f}")
        self.logger.info(f"  Predicted no-shows (TP): {tp}")
        self.logger.info(f"  Estimated prevented: {prevented_noshows}")
        self.logger.info(f"  Estimated net savings: ${impact['net_savings']:,.0f}")
        
        return impact
    
    # ==================== VISUALIZATION METHODS ====================
    
    def plot_roc_curves(
        self, 
        save: bool = True,
        filename: str = "roc_curves.png"
    ) -> plt.Figure:
        """
        Plot ROC curves for all evaluated models.
        
        Parameters
        ----------
        save : bool
            Whether to save the figure
        filename : str
            Output filename
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.roc_curves)))
        
        for (name, (fpr, tpr, _)), color in zip(self.roc_curves.items(), colors):
            auc = self.results[name].roc_auc
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{name} (AUC = {auc:.3f})')
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved ROC curves to {filepath}")
        
        return fig
    
    def plot_precision_recall_curves(
        self, 
        save: bool = True,
        filename: str = "precision_recall_curves.png"
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for all evaluated models.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.pr_curves)))
        
        for (name, (precision, recall, _)), color in zip(self.pr_curves.items(), colors):
            ap = self.results[name].average_precision
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{name} (AP = {ap:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved PR curves to {filepath}")
        
        return fig
    
    def plot_confusion_matrices(
        self, 
        save: bool = True,
        filename: str = "confusion_matrices.png"
    ) -> plt.Figure:
        """
        Plot confusion matrices for all evaluated models.
        """
        n_models = len(self.results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            
            cm = result.confusion_matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=ax,
                xticklabels=['Showed Up', 'No-Show'],
                yticklabels=['Showed Up', 'No-Show'],
                cbar=False
            )
            
            ax.set_title(f'{name}\nAccuracy: {result.accuracy:.3f}', fontsize=11)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved confusion matrices to {filepath}")
        
        return fig
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str,
        top_n: int = 20,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with 'feature' and 'importance' columns
        model_name : str
            Name of the model
        top_n : int
            Number of top features to show
        save : bool
            Whether to save the figure
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / f"feature_importance_{model_name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved feature importance to {filepath}")
        
        return fig
    
    def plot_metrics_comparison(
        self,
        save: bool = True,
        filename: str = "metrics_comparison.png"
    ) -> plt.Figure:
        """
        Plot bar chart comparing metrics across models.
        """
        comparison_df = self.get_comparison_table()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.15
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - len(metrics)/2 + 0.5)
            ax.bar(x + offset, comparison_df[metric], width, 
                  label=metric, color=color, edgecolor='white')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.figures_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved metrics comparison to {filepath}")
        
        return fig
    
    def generate_all_plots(self) -> None:
        """Generate all evaluation plots."""
        self.logger.info("Generating all evaluation plots...")
        
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_confusion_matrices()
        self.plot_metrics_comparison()
        
        self.logger.info("All plots generated!")
    
    def save_results(
        self, 
        output_dir: Union[str, Path]
    ) -> None:
        """
        Save all evaluation results.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        comparison_df = self.get_comparison_table()
        comparison_df.to_csv(output_dir / "evaluation_results.csv", index=False)
        
        # Save detailed results as JSON
        results_dict = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {output_dir}")