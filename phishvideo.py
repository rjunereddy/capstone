#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VIDEO PHISHING DETECTION SYSTEM - PRODUCTION READY
- Works with Celeb-DF-v2 (Deepfake) datasets
- Anomaly-based detection (learns deepfake/tampering patterns)
- Fusion-ready with embeddings
- Robust error handling
- Interactive testing
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import time
import random
import glob
import cv2
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# Machine Learning
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import joblib

print("=" * 100)
print("🎥 VIDEO PHISHING DETECTION - PRODUCTION READY")
print("=" * 100)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# =========================
# CONFIGURATION
# =========================
BASE_PATH = "G:/My Drive/Phishguard"
MODEL_SAVE_PATH = f"{BASE_PATH}/video_phishing_detector"
EMBEDDING_DIM = 64
MAX_SAMPLES = 3000  # Reduced due to video processing intensity
RANDOM_SEED = 42

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# VIDEO PATHS (Celeb-DF-v2)
# =========================
VIDEO_PATHS = [
    "G:/My Drive/Phishguard/Celeb-DF-v2/Celeb-synthesis", # Primary training (Deepfake)
    "G:/My Drive/Phishguard/Celeb-DF-v2/Celeb-real",
    "G:/My Drive/Phishguard/Celeb-DF-v2/YouTube-real"
]

# =========================
# FEATURE EXTRACTOR
# =========================
class VideoFeatureExtractor:
    """Extracts visual and temporal features for anomaly detection"""
    
    def __init__(self, num_frames_to_sample=15):
        self.num_frames_to_sample = num_frames_to_sample
    
    def extract_features(self, video_path):
        """Extract features from a single video file"""
        features = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._zero_features()
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Basic validation
            if fps <= 0 or frame_count <= 0:
                fps = 30
                frame_count = 100
                
            features['fps'] = fps
            features['resolution_score'] = (width * height) / 1000000.0  # Megapixels
            features['total_frames'] = frame_count
            features['duration'] = frame_count / fps if fps > 0 else 0
            
            # Sample frames uniformly across the video
            frame_indices = np.linspace(0, max(0, frame_count - 1), self.num_frames_to_sample, dtype=int)
            
            laplacian_vars = []
            brightnesses = []
            contrasts = []
            diffs = []
            color_variances_h = []
            color_variances_s = []
            
            prev_gray = None
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                    
                # Downscale frame to speed up processing
                small_frame = cv2.resize(frame, (320, 240))
                
                # Spatial and Blur Features (Gray)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                laplacian_vars.append(lap_var)
                brightnesses.append(np.mean(gray))
                contrasts.append(np.std(gray))
                
                # Color statistics (HSV)
                hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                color_variances_h.append(np.var(h))
                color_variances_s.append(np.var(s))
                
                # Temporal Features (Frame Difference to detect splice/jitter)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    diffs.append(np.mean(diff))
                    
                prev_gray = gray
                
            cap.release()
            
            # Helper to add stats
            def add_stats(name, values):
                if not values:
                    features[f'{name}_mean'] = 0
                    features[f'{name}_std'] = 0
                    features[f'{name}_max'] = 0
                else:
                    features[f'{name}_mean'] = np.mean(values)
                    features[f'{name}_std'] = np.std(values)
                    features[f'{name}_max'] = np.max(values)
                    
            add_stats('laplacian', laplacian_vars)
            add_stats('brightness', brightnesses)
            add_stats('contrast', contrasts)
            add_stats('frame_diff', diffs)
            add_stats('color_h_var', color_variances_h)
            add_stats('color_s_var', color_variances_s)
            
        except Exception as e:
            return self._zero_features()
            
        return features
    
    def _zero_features(self):
        """Return zero features for failed files"""
        features = {}
        features['fps'] = 0
        features['resolution_score'] = 0
        features['total_frames'] = 0
        features['duration'] = 0
        
        feature_names = ['laplacian', 'brightness', 'contrast', 'frame_diff', 'color_h_var', 'color_s_var']
        for name in feature_names:
            features[f'{name}_mean'] = 0
            features[f'{name}_std'] = 0
            features[f'{name}_max'] = 0
            
        return features
    
    def extract_batch(self, video_files, verbose=True):
        """Extract features for multiple video files"""
        features_list = []
        iterator = tqdm(video_files, desc="Extracting video features") if verbose else video_files
        
        for video_file in iterator:
            features = self.extract_features(video_file)
            features_list.append(features)
            
        return pd.DataFrame(features_list)

# =========================
# DATA LOADER
# =========================
class VideoDataLoader:
    """Loads video files from all paths with random sampling"""
    
    def __init__(self, video_paths, max_samples=MAX_SAMPLES):
        self.video_paths = video_paths
        self.max_samples = max_samples
    
    def find_deepfake_files(self):
        """Find specifically the synthesized/fake videos for training the anomaly detector"""
        all_files = []
        
        print("\n📂 Scanning for Deepfake video files for training...")
        # Target the synthesis folder specifically for Isolation Forest target patterns
        target_paths = [p for p in self.video_paths if "synthesis" in p.lower() or "fake" in p.lower()]
        if not target_paths:
            print("   ⚠️ Proceeding with all available paths as no explicit 'synthesis' folder found.")
            target_paths = self.video_paths
            
        for path in target_paths:
            if not os.path.exists(path):
                print(f"   ⚠️ Path not found: {path}")
                continue
                
            mp4_files = glob.glob(os.path.join(path, "*.mp4"))
            avi_files = glob.glob(os.path.join(path, "*.avi"))
            
            files = mp4_files + avi_files
            all_files.extend(files)
            print(f"   ✓ {os.path.basename(path)}: {len(files)} files")
            
        all_files = list(set(all_files))
        print(f"\n   Total unique fake video files: {len(all_files)}")
        return all_files
    
    def load_random_sample(self):
        """Load random sample of video files for training"""
        all_files = self.find_deepfake_files()
        
        if not all_files:
            print("❌ No deepfake video files found!")
            return []
            
        sample_size = min(self.max_samples, len(all_files))
        sampled_files = random.sample(all_files, sample_size)
        
        print(f"\n🎲 Randomly sampled {len(sampled_files)} files for training")
        return sampled_files

# =========================
# ANOMALY DETECTOR
# =========================
class VideoAnomalyDetector:
    """Detects fake/deepfake video using anomaly detection"""
    
    def __init__(self):
        self.feature_extractor = VideoFeatureExtractor()
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=EMBEDDING_DIM)
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
    def train(self, video_files, contamination=0.1):
        """
        Train on DEEPFAKE video files
        Uses Isolation Forest for anomaly detection
        """
        print("\n" + "=" * 80)
        print("🏋️ TRAINING VIDEO ANOMALY DETECTOR")
        print("=" * 80)
        print(f"Training on {len(video_files)} fake video files")
        
        print("\n🔧 Extracting features...")
        X = self.feature_extractor.extract_batch(video_files, verbose=True)
        
        self.feature_names = X.columns.tolist()
        print(f"   Features extracted: {X.shape[1]}")
        
        X = X.fillna(0)
        
        print("   Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print("   Generating embeddings...")
        X_embedding = self.pca.fit_transform(X_scaled)
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"   PCA explained variance: {explained_var:.2%}")
        
        print("\n🚀 Training Isolation Forest...")
        self.model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_SEED,
            n_estimators=200,
            n_jobs=-1
        )
        
        start_time = time.time()
        self.model.fit(X_scaled)
        training_time = time.time() - start_time
        
        predictions = self.model.predict(X_scaled)
        anomaly_count = np.sum(predictions == -1)
        
        print(f"\n📊 Training Results:")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Anomalies detected in training: {anomaly_count}/{len(X_scaled)} ({anomaly_count/len(X_scaled)*100:.1f}%)")
        
        self.is_trained = True
        return True
        
    def analyze_video(self, video_path):
        """
        Analyze video file - returns risk score (0-1)
        Higher score = more likely to be FAKE/DEEPFAKE
        """
        if not self.is_trained:
            return self._fallback_analysis(video_path)
            
        if not os.path.exists(video_path):
            return {
                'modality': 'video', 'risk_score': 0.5, 'error': 'File not found', 'video_path': video_path
            }
            
        features = self.feature_extractor.extract_features(video_path)
        X = pd.DataFrame([features])
        X = X.fillna(0)
        
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        
        anomaly_score = self.model.decision_function(X_scaled)[0]
        # Normalize to 0-1 range, inverted so anomaly -> high risk
        risk_score = 1 / (1 + np.exp(-anomaly_score)) 
        risk_score = 1 - risk_score
        
        X_pca = self.pca.transform(X_scaled)
        embedding = X_pca[0] if len(X_pca) > 0 else np.zeros(EMBEDDING_DIM)
        
        if risk_score >= 0.8:
            risk_level, risk_text, rec = "🔴 CRITICAL", "CRITICAL", "⚠️ Video shows strong signs of manipulation!"
        elif risk_score >= 0.6:
            risk_level, risk_text, rec = "🟠 HIGH", "HIGH", "⚠️ High risk - temporal/spatial inconsistencies detected"
        elif risk_score >= 0.4:
            risk_level, risk_text, rec = "🟡 MEDIUM", "MEDIUM", "⚠️ Medium risk - slight anomalies present"
        elif risk_score >= 0.2:
            risk_level, risk_text, rec = "🔵 LOW", "LOW", "✅ Low risk - likely legitimate"
        else:
            risk_level, risk_text, rec = "🟢 SAFE", "SAFE", "✅ Video appears to be completely untouched"
            
        return {
            'modality': 'video',
            'risk_score': risk_score,
            'safe_score': 1 - risk_score,
            'embedding': embedding,
            'confidence': abs(risk_score - 0.5) * 2,
            'risk_level': risk_level,
            'risk_text': risk_text,
            'is_phishing': risk_score > 0.5,
            'video_path': video_path,
            'recommendation': rec,
            'timestamp': datetime.now().isoformat(),
            'features': {
                'Laplacian_Blur': features.get('laplacian_mean', 0),
                'Frame_Jitter': features.get('frame_diff_mean', 0),
                'Duration_Sec': features.get('duration', 0)
            }
        }
        
    def _fallback_analysis(self, video_path):
        return {
            'modality': 'video', 'risk_score': 0.5, 'safe_score': 0.5, 'embedding': np.zeros(EMBEDDING_DIM),
            'confidence': 0, 'risk_level': "⚠️ MODEL NOT TRAINED", 'risk_text': "UNKNOWN",
            'is_phishing': False, 'video_path': video_path, 'recommendation': "Please train the model first",
            'features': {}, 'timestamp': datetime.now().isoformat()
        }
        
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, 'model.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(path, 'pca.pkl'))
        joblib.dump(self.feature_names, os.path.join(path, 'feature_names.pkl'))
        print(f"\n✓ Model saved to {path}")
        
    def load(self, path):
        self.model = joblib.load(os.path.join(path, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        self.pca = joblib.load(os.path.join(path, 'pca.pkl'))
        self.feature_names = joblib.load(os.path.join(path, 'feature_names.pkl'))
        self.is_trained = True
        print(f"✓ Model loaded from {path}")

# =========================
# INTERACTIVE TESTER
# =========================
class VideoTester:
    def __init__(self, detector):
        self.detector = detector
        
    def run(self):
        print("\n" + "=" * 100)
        print("🎯 INTERACTIVE VIDEO TESTER")
        print("=" * 100)
        print("\nCommands:")
        print("  • Enter video file path to test")
        print("  • 'quit' - Exit")
        print("-" * 100)
        
        while True:
            user_input = input("\n🎥 Enter video file path: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']: break
            if user_input: self._test_video(user_input)
            
    def _test_video(self, video_path):
        # Removing quotes for dragged files
        video_path = video_path.replace('"', '')
        if not os.path.exists(video_path):
            print(f"❌ File not found: {video_path}")
            return
            
        result = self.detector.analyze_video(video_path)
        
        print("\n" + "=" * 80)
        print("📊 VIDEO ANALYSIS RESULT")
        print("=" * 80)
        print(f"File: {result['video_path']}")
        print(f"\nRisk Score: {result['risk_score']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Classification: {'⚠️ DEEPFAKE/TAMPERED' if result['is_phishing'] else '✅ REAL'}")
        
        bar_length = 40
        filled = int(bar_length * result['risk_score'])
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\nRisk Meter: [{bar}]")
        print(f"Fusion Modality: {result['modality']} | Embedding dims: {len(result['embedding'])}")
        print("=" * 80)

# =========================
# MAIN
# =========================
def main():
    print("\n🚀 Starting Video Phishing Detection System")
    
    if os.path.exists(os.path.join(MODEL_SAVE_PATH, 'model.pkl')):
        print(f"\n✅ Found existing model")
        response = input("Use existing model? (y/n): ").strip().lower()
        if response == 'y':
            detector = VideoAnomalyDetector()
            detector.load(MODEL_SAVE_PATH)
            tester = VideoTester(detector)
            tester.run()
            return
            
    loader = VideoDataLoader(VIDEO_PATHS, MAX_SAMPLES)
    video_files = loader.load_random_sample()
    
    if not video_files:
        print("\n❌ No videos found! Verify paths or proceed without training.")
        return
        
    detector = VideoAnomalyDetector()
    detector.train(video_files)
    detector.save(MODEL_SAVE_PATH)
    
    tester = VideoTester(detector)
    tester.run()

if __name__ == "__main__":
    main()
