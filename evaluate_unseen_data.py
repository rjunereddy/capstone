import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report

from train_models import extract_feature_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'url_model')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_FILE = os.path.join(BASE_DIR, 'unseen_test_data.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_models():
    print(f"Loading pre-trained models from {MODELS_DIR}...")
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        selector = joblib.load(os.path.join(MODELS_DIR, 'selector.joblib'))
        model = joblib.load(os.path.join(MODELS_DIR, 'model.joblib'))
        return scaler, selector, model
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Dataset {DATA_FILE} not found. Please run get_test_dataset.py first.")
        sys.exit(1)

    print(f"Loading unseen dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    if 'url' not in df.columns or 'label' not in df.columns:
        print("Dataset must contain 'url' and 'label' columns.")
        sys.exit(1)

    urls = df['url'].tolist()
    y_true = df['label'].tolist()
    
    print(f"Dataset size: {len(urls)} URLs")
    print("Extracting features (this may take a moment)...")
    
    t0 = time.time()
    X_df = extract_feature_matrix(urls, verbose=True)
    print(f"Feature extraction took {time.time() - t0:.2f} seconds.")
    
    scaler, selector, model = load_models()
    
    print("Applying preprocessing steps...")
    X_scaled = scaler.transform(X_df)
    X_selected = selector.transform(X_scaled)
    
    print("Making predictions on unseen data...")
    y_pred = model.predict(X_selected)
    y_prob = model.predict_proba(X_selected)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print("   PERFORMANCE ON UNSEEN DATA (TEST SPLIT)")
    print("="*50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Phishing"]))
    
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
    plt.title('Confusion Matrix on Unseen Data')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_test.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix plot to: {cm_path}")
    
    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Unseen Data')
    plt.legend(loc="lower right")
    roc_path = os.path.join(RESULTS_DIR, 'roc_curve_test.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curve plot to: {roc_path}")
    
    print("\nDone! Ready for teacher presentation.")

if __name__ == "__main__":
    main()
