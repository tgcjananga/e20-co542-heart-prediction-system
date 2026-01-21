import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
import os

# Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("HEART DISEASE PREDICTION - MODEL TESTING")
print("=" * 60)

# Load the test data
print("\n[1/6] Loading dataset...")
data = pd.read_csv("data/heart_disease.csv")
X = data.drop(columns=["target"])
y = data["target"]

# Split dataset (same split as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Test set size: {len(X_test)} samples")

# Load the trained model and scaler
print("\n[2/6] Loading trained model and scaler...")
model = joblib.load("model/xgboost_model.pkl")
scaler = joblib.load("model/scaler.pkl")
print("✓ Model and scaler loaded successfully")

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Make predictions
print("\n[3/6] Making predictions...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
print("✓ Predictions complete")

# Calculate metrics
print("\n[4/6] Calculating performance metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print metrics
print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("=" * 60)

# Detailed classification report
print("\n[5/6] Generating detailed classification report...")
print("\nCLASSIFICATION REPORT:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# Save metrics to file
metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_df.to_csv(f"{RESULTS_DIR}/model_metrics.csv", index=False)
print(f"\n✓ Metrics saved to {RESULTS_DIR}/model_metrics.csv")

# Generate visualizations
print("\n[6/6] Generating visualizations...")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")

# 2. ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ ROC curve saved to {RESULTS_DIR}/roc_curve.png")

# 3. Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_vals, precision_vals, color='blue', lw=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Precision-Recall curve saved to {RESULTS_DIR}/precision_recall_curve.png")

# 4. Feature Importance
feature_names = X.columns.tolist()
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Feature importance saved to {RESULTS_DIR}/feature_importance.png")

# Save feature importance to CSV
feature_importance_df.to_csv(f"{RESULTS_DIR}/feature_importance.csv", index=False)
print(f"✓ Feature importance data saved to {RESULTS_DIR}/feature_importance.csv")

# 5. Metrics Comparison Bar Chart
plt.figure(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
plt.ylim([0, 1])
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Metrics comparison saved to {RESULTS_DIR}/metrics_comparison.png")

# Print confusion matrix details
print("\nCONFUSION MATRIX BREAKDOWN:")
print("-" * 60)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives:  {tn} (Correctly predicted No Disease)")
print(f"False Positives: {fp} (Incorrectly predicted Disease)")
print(f"False Negatives: {fn} (Missed Disease cases)")
print(f"True Positives:  {tp} (Correctly predicted Disease)")
print("-" * 60)

# Calculate additional metrics
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\nSpecificity (True Negative Rate): {specificity * 100:.2f}%")
print(f"Sensitivity (True Positive Rate): {sensitivity * 100:.2f}%")

# Summary report
print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)
print(f"\nAll results saved in '{RESULTS_DIR}/' directory:")
print(f"  • model_metrics.csv")
print(f"  • confusion_matrix.png")
print(f"  • roc_curve.png")
print(f"  • precision_recall_curve.png")
print(f"  • feature_importance.png")
print(f"  • feature_importance.csv")
print(f"  • metrics_comparison.png")
print("\n" + "=" * 60)