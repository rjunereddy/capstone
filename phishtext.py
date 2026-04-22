#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TEXT MODULE - Phishing Detection with Embeddings for Fusion
Based on your working code but adds embedding extraction
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

print("=" * 80)
print("🔒 TEXT MODULE - PHISHING DETECTION WITH EMBEDDINGS")
print("=" * 80)
print("Output: Risk Score + 768-dim Embedding Vector (Ready for Fusion)")
print("=" * 80)

# =========================
# CONFIGURATION
# =========================
BASE_PATH = "G:/My Drive/Phishguard"
MODEL_SAVE_PATH = f"{BASE_PATH}/text_model_final"

# =========================
# DATA LOADING FUNCTION (PRESERVED FROM YOUR CODE)
# =========================
def load_all_data():
    """Load and combine all datasets"""
    print("\n📂 Loading datasets...")
    all_data = []
    
    # Define files to load
    files_to_load = [
        ("CEAS_08.csv", ['body', 'subject']),
        ("Enron.csv", ['body', 'subject']),
        ("Ling.csv", ['body', 'subject']),
        ("Nazario.csv", ['body', 'subject']),
        ("Nigerian_Fraud.csv", ['body', 'subject']),
        ("SpamAssasin.csv", ['body', 'subject']),
        ("phishing_email.csv", ['text_combined']),
    ]
    
    # Load each dataset
    for filename, text_cols in files_to_load:
        file_path = os.path.join(BASE_PATH, "archive", filename)
        
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {filename}")
            continue
            
        try:
            # Try different encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except:
                    continue
            
            if df is None:
                print(f"✗ Could not load {filename}")
                continue
                
            print(f"✓ Loaded {filename}: {len(df)} rows")
            
            # Combine text columns
            text_parts = []
            for col in text_cols:
                if col in df.columns:
                    text_parts.append(df[col].fillna('').astype(str))
            
            if text_parts:
                df['combined_text'] = text_parts[0]
                for part in text_parts[1:]:
                    df['combined_text'] = df['combined_text'] + " " + part
            else:
                print(f"  ⚠️ No text columns found in {filename}")
                continue
            
            # Handle labels
            if 'label' in df.columns:
                # Map labels
                def map_label(x):
                    x = str(x).lower().strip()
                    if x in ['ham', 'safe', 'legitimate', '0', '0.0', 'not spam']:
                        return 0
                    elif x in ['spam', 'phishing', '1', '1.0', 'phish', 'fraud', 'scam']:
                        return 1
                    else:
                        return None
                
                df['label'] = df['label'].apply(map_label)
                df = df.dropna(subset=['label'])
                df['label'] = df['label'].astype(int)
                
                # Create final dataframe
                temp_df = pd.DataFrame({
                    'text': df['combined_text'],
                    'label': df['label'],
                    'source': filename
                })
                all_data.append(temp_df)
                print(f"  📊 Labels - Safe: {sum(temp_df['label']==0)}, Phishing: {sum(temp_df['label']==1)}")
            else:
                print(f"  ⚠️ No labels in {filename}, skipping")
                
        except Exception as e:
            print(f"✗ Error loading {filename}: {str(e)[:100]}")
    
    # Load SMS dataset
    sms_path = os.path.join(BASE_PATH, "sms+spam+collection", "SMSSpamCollection")
    if os.path.exists(sms_path):
        try:
            sms_df = pd.read_csv(sms_path, sep='\t', header=None, names=['label', 'text'], encoding='utf-8')
            label_mapping = {'ham': 0, 'spam': 1}
            sms_df['label'] = sms_df['label'].map(label_mapping)
            sms_df = sms_df.dropna()
            sms_df['source'] = 'SMSSpamCollection'
            all_data.append(sms_df[['text', 'label', 'source']])
            print(f"✓ Loaded SMS dataset: {len(sms_df)} samples")
        except Exception as e:
            print(f"✗ Error loading SMS dataset: {e}")
    
    if not all_data:
        print("\n⚠️ No data found! Creating sample data for testing...")
        return create_sample_data()
    
    # Combine all datasets
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'], keep='first')
    
    print(f"\n✅ Total combined dataset: {len(combined)} samples")
    print(f"   Safe: {sum(combined['label']==0)} ({sum(combined['label']==0)/len(combined)*100:.1f}%)")
    print(f"   Phishing: {sum(combined['label']==1)} ({sum(combined['label']==1)/len(combined)*100:.1f}%)")
    
    return combined

def create_sample_data():
    """Create sample data for testing"""
    safe_texts = [
        "Let's meet for lunch tomorrow at 2pm",
        "Please find the attached quarterly report for review",
        "Thanks for your email, I'll get back to you soon",
    ] * 30
    
    phish_texts = [
        "URGENT: Your bank account has been compromised! Verify now: http://fake-bank.com",
        "Congratulations! You've won $1,000,000! Click here to claim your prize",
        "Your PayPal account has been limited. Update payment info immediately",
    ] * 20
    
    texts = safe_texts + phish_texts
    labels = [0] * len(safe_texts) + [1] * len(phish_texts)
    
    df = pd.DataFrame({'text': texts, 'label': labels, 'source': 'sample'})
    print(f"📊 Created sample data: {len(df)} samples")
    return df

# =========================
# TEXT CLEANING (PRESERVED)
# =========================
def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.!?]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Limit length for memory efficiency
    if len(text) > 5000:
        text = text[:5000]
    return text

# =========================
# DATASET CLASS (PRESERVED)
# =========================
class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# =========================
# TEXT DETECTOR CLASS (WITH EMBEDDINGS)
# =========================
class TextPhishingDetector:
    """
    Text phishing detector that outputs:
    - risk_score: float (0-1)
    - embedding: numpy array (768-dim) for fusion
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, texts, labels, epochs=2, batch_size=16):
        """Train the model (preserves all your training logic)"""
        print("\n🏋️ Training Text Model...")
        
        # Clean texts
        cleaned_texts = [clean_text(t) for t in texts]
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            cleaned_texts, labels, test_size=0.3, stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        print(f"   Training: {len(X_train)}")
        print(f"   Validation: {len(X_val)}")
        print(f"   Test: {len(X_test)}")
        
        # Tokenize
        print("\n🔤 Tokenizing...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        def tokenize_texts(texts):
            return self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors=None)
        
        train_enc = tokenize_texts(X_train)
        val_enc = tokenize_texts(X_val)
        test_enc = tokenize_texts(X_test)
        
        # Create datasets
        train_dataset = PhishingDataset(train_enc, y_train)
        val_dataset = PhishingDataset(val_enc, y_val)
        test_dataset = PhishingDataset(test_enc, y_test)
        
        # Load model (with hidden states for embeddings)
        print("\n🤖 Loading DistilBERT...")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=2,
            output_hidden_states=True  # CRITICAL: Need embeddings for fusion
        )
        self.model.to(self.device)
        
        # Training arguments (preserved from your code)
        training_args = TrainingArguments(
            output_dir='./text_model_results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=(self.device == 'cuda'),
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            logging_first_step=True
        )
        
        # Metrics function
        def compute_metrics(p):
            preds = p.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
            acc = accuracy_score(p.label_ids, preds)
            return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        print("\n🚀 Training...")
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate(test_dataset)
        print(f"\n📊 Test Results:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
        
        self.is_trained = True
        return results
    
    def predict(self, text, threshold=0.5):
        """
        Predict phishing with embedding for fusion
        Returns: {
            'modality': 'text',
            'risk_score': float,
            'embedding': np.array (768-dim),
            'confidence': float,
            'risk_level': str,
            'is_phishing': bool
        }
        """
        if not self.is_trained or self.model is None:
            print("⚠️ Model not trained! Using fallback.")
            return self._fallback_prediction(text)
        
        cleaned = clean_text(text)
        
        # Tokenize
        inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, 
                               padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get probability score
            probs = torch.softmax(outputs.logits, dim=1)
            phishing_score = probs[0][1].item()
            safe_score = probs[0][0].item()
            
            # Get embedding (average of last hidden states) - FOR FUSION
            hidden_states = outputs.hidden_states[-1]  # Last layer: [batch, seq_len, hidden_dim]
            embedding = hidden_states.mean(dim=1).cpu().numpy()[0]  # 768-dim vector
            
            # Calculate confidence
            confidence = abs(phishing_score - 0.5) * 2
            
            # Risk level
            if phishing_score >= 0.8:
                risk_level = "🔴 CRITICAL"
            elif phishing_score >= 0.6:
                risk_level = "🟠 HIGH"
            elif phishing_score >= 0.4:
                risk_level = "🟡 MEDIUM"
            elif phishing_score >= 0.2:
                risk_level = "🔵 LOW"
            else:
                risk_level = "🟢 SAFE"
            
            return {
                'modality': 'text',
                'risk_score': phishing_score,
                'safe_score': safe_score,
                'embedding': embedding,  # For correlation in fusion
                'confidence': confidence,
                'risk_level': risk_level,
                'is_phishing': phishing_score > threshold,
                'cleaned_text': cleaned[:200]
            }
    
    def predict_batch(self, texts):
        """Predict for multiple texts"""
        return [self.predict(t) for t in texts]
    
    def _fallback_prediction(self, text):
        """Fallback when model not trained"""
        phishing_keywords = ['urgent', 'verify', 'account', 'click', 'bank', 
                            'password', 'suspend', 'compromised', 'win', 'prize']
        
        cleaned = clean_text(text)
        words = cleaned.split()
        score = sum(1 for kw in phishing_keywords if kw in words) / len(phishing_keywords)
        score = min(score, 0.95)
        
        # Random embedding (fallback)
        embedding = np.random.randn(768) * 0.1
        
        return {
            'modality': 'text',
            'risk_score': score,
            'safe_score': 1 - score,
            'embedding': embedding,
            'confidence': abs(score - 0.5) * 2,
            'risk_level': "HIGH" if score > 0.5 else "LOW",
            'is_phishing': score > 0.5,
            'cleaned_text': cleaned[:200]
        }
    
    def save(self, path):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        """Load model and tokenizer"""
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            path, 
            output_hidden_states=True
        )
        self.model.to(self.device)
        self.is_trained = True
        print(f"✓ Model loaded from {path}")

# =========================
# MAIN TRAINING FUNCTION
# =========================
def main():
    """Main training function (preserves all your code structure)"""
    
    # Load data
    data = load_all_data()
    
    # Clean text
    print("\n🧹 Cleaning text data...")
    data['clean_text'] = data['text'].apply(clean_text)
    data = data[data['clean_text'].str.len() > 10]
    data = data[data['clean_text'].str.len() < 5000]
    print(f"✅ After cleaning: {len(data)} samples")
    
    # Basic EDA
    print("\n📊 Exploratory Data Analysis:")
    print(f"   Average text length: {data['clean_text'].str.len().mean():.0f} chars")
    print(f"   Safe: {sum(data['label']==0)}, Phishing: {sum(data['label']==1)}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
    axes[0].set_title('Class Distribution')
    axes[0].set_xlabel('Label (0=Safe, 1=Phishing)')
    
    axes[1].hist([data[data['label']==0]['clean_text'].str.len(), 
                  data[data['label']==1]['clean_text'].str.len()], 
                 bins=50, alpha=0.7, label=['Safe', 'Phishing'])
    axes[1].set_title('Text Length Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=100)
    plt.close()
    print("📈 EDA plot saved as 'eda_analysis.png'")
    
    # Prepare data for training
    X = data['clean_text'].tolist()
    y = data['label'].tolist()
    
    # Sample if too large
    if len(X) > 50000:
        print(f"\n📊 Dataset large ({len(X)}), sampling 50,000 for faster training...")
        from sklearn.utils import resample
        X, y = resample(X, y, n_samples=50000, random_state=42, stratify=y)
    
    # Create and train detector
    detector = TextPhishingDetector()
    results = detector.train(X, y, epochs=2, batch_size=16)
    
    # Save model
    detector.save(MODEL_SAVE_PATH)
    
    # =========================
    # DEMO WITH EMBEDDINGS (FOR FUSION)
    # =========================
    print("\n" + "=" * 80)
    print("🎯 DEMO: Text Module with Fusion-Ready Outputs")
    print("=" * 80)
    
    test_cases = [
        "URGENT: Your PayPal account has been suspended! Verify now: http://paypal-secure.com",
        "Hey, are we still meeting for coffee tomorrow at 3pm?",
        "Congratulations! You've won a free iPhone! Click here to claim: http://win-iphone.com",
        "Please find the quarterly report attached for your review",
    ]
    
    print("\n🔍 Testing with Embeddings (Ready for Fusion):\n")
    
    for text in test_cases:
        result = detector.predict(text)
        
        print("=" * 70)
        print(f"📝 Text: {text[:80]}...")
        print(f"\n📊 Output for Fusion Module:")
        print(f"   ├─ modality: '{result['modality']}'")
        print(f"   ├─ risk_score: {result['risk_score']:.4f}")
        print(f"   ├─ embedding: [{result['embedding'][0]:.4f}, {result['embedding'][1]:.4f}, ...] (768-dim)")
        print(f"   ├─ confidence: {result['confidence']:.4f}")
        print(f"   ├─ risk_level: {result['risk_level']}")
        print(f"   └─ is_phishing: {result['is_phishing']}")
        
        # Visual bar
        bar_length = 30
        filled = int(bar_length * result['risk_score'])
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\n   Risk Meter: [{bar}] {result['risk_score']:.1%}")
        print()
    
    print("=" * 80)
    print("✅ TEXT MODULE READY FOR FUSION")
    print("=" * 80)
    print(f"📁 Model saved: {MODEL_SAVE_PATH}")
    print(f"📊 Output: risk_score (0-1) + embedding (768-dim)")
    print("\n🎯 How to use with your fusion module:")
    print("""
    # In your fusion code:
    text_output = text_detector.predict(email_content)
    
    # Now you have:
    text_output['risk_score']   # 0.87
    text_output['embedding']    # 768-dim vector for correlation
    
    # Pass to fusion:
    all_modalities = [text_output, url_output, image_output]
    final_risk = correlation_fusion(all_modalities)
    """)

if __name__ == "__main__":
    main()