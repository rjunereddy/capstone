#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AUDIO PHISHING DETECTION SYSTEM - PRODUCTION READY
- Works with ASVspoof 2019 & 2021 datasets
- Anomaly-based detection (learns spoof patterns)
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
import soundfile as sf
import librosa
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# Machine Learning
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import joblib

print("=" * 100)
print("🎵 AUDIO PHISHING DETECTION - PRODUCTION READY")
print("=" * 100)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# =========================
# CONFIGURATION
# =========================
BASE_PATH = "G:/My Drive/Phishguard"
MODEL_SAVE_PATH = f"{BASE_PATH}/audio_phishing_detector"
EMBEDDING_DIM = 64
SAMPLE_RATE = 16000
DURATION = 4  # seconds
MAX_SAMPLES = 30000  # Limit for training (random sampling)
RANDOM_SEED = 42

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# AUDIO PATHS
# =========================
AUDIO_PATHS = [
    "G:/My Drive/Phishguard/LA/LA/ASVspoof2019_LA_dev/flac",
    "G:/My Drive/Phishguard/LA/LA/ASVspoof2019_LA_train",
    "G:/My Drive/Phishguard/ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac",
    "G:/My Drive/Phishguard/PA/ASVspoof2019_PA_dev/flac",
]

# =========================
# FEATURE EXTRACTOR
# =========================
class AudioFeatureExtractor:
    """Extracts acoustic features for anomaly detection"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration
    
    def extract_features(self, audio_path):
        """Extract features from a single audio file"""
        features = {}
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Handle silent files
            if len(audio) == 0 or np.max(np.abs(audio)) < 0.01:
                return self._zero_features()
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Pad or truncate to fixed length
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)))
            else:
                audio = audio[:self.target_length]
            
            # ========== MFCC Features (20 features) ==========
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            for i in range(20):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
                features[f'mfcc_{i}_skew'] = skew(mfccs[i])
                features[f'mfcc_{i}_kurt'] = kurtosis(mfccs[i])
            
            # ========== Spectral Features ==========
            spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spec_centroid_mean'] = np.mean(spec_cent)
            features['spec_centroid_std'] = np.std(spec_cent)
            
            spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features['spec_bw_mean'] = np.mean(spec_bw)
            features['spec_bw_std'] = np.std(spec_bw)
            
            spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['spec_rolloff_mean'] = np.mean(spec_rolloff)
            features['spec_rolloff_std'] = np.std(spec_rolloff)
            
            # ========== Zero Crossing Rate ==========
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # ========== RMS Energy ==========
            rms = librosa.feature.rms(y=audio)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # ========== Pitch Features ==========
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            # ========== Spectral Contrast ==========
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            for i in range(min(7, contrast.shape[0])):
                features[f'contrast_{i}_mean'] = np.mean(contrast[i])
                features[f'contrast_{i}_std'] = np.std(contrast[i])
            
            # ========== Chroma Features ==========
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            for i in range(12):
                features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                features[f'chroma_{i}_std'] = np.std(chroma[i])
            
            # ========== Additional Features ==========
            features['duration'] = len(audio) / sr
            features['energy'] = np.sum(audio**2) / len(audio)
            
            # Entropy
            spec = np.abs(librosa.stft(audio))
            spec_norm = spec / (np.sum(spec) + 1e-12)
            features['spectral_entropy'] = -np.sum(spec_norm * np.log2(spec_norm + 1e-12))
            
            # Harmonic-to-Noise Ratio (simplified)
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if len(autocorr) > 1:
                features['hnr'] = 10 * np.log10(np.max(autocorr[1:]) / (np.var(audio) + 1e-12))
                features['hnr'] = np.clip(features['hnr'], 0, 40)
            else:
                features['hnr'] = 0
            
        except Exception as e:
            return self._zero_features()
        
        return features
    
    def _zero_features(self):
        """Return zero features for failed files"""
        features = {}
        # Add all expected feature names with zero values
        for i in range(20):
            features[f'mfcc_{i}_mean'] = 0
            features[f'mfcc_{i}_std'] = 0
            features[f'mfcc_{i}_skew'] = 0
            features[f'mfcc_{i}_kurt'] = 0
        
        features['spec_centroid_mean'] = 0
        features['spec_centroid_std'] = 0
        features['spec_bw_mean'] = 0
        features['spec_bw_std'] = 0
        features['spec_rolloff_mean'] = 0
        features['spec_rolloff_std'] = 0
        features['zcr_mean'] = 0
        features['zcr_std'] = 0
        features['rms_mean'] = 0
        features['rms_std'] = 0
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        
        for i in range(7):
            features[f'contrast_{i}_mean'] = 0
            features[f'contrast_{i}_std'] = 0
        
        for i in range(12):
            features[f'chroma_{i}_mean'] = 0
            features[f'chroma_{i}_std'] = 0
        
        features['duration'] = 0
        features['energy'] = 0
        features['spectral_entropy'] = 0
        features['hnr'] = 0
        
        return features
    
    def extract_batch(self, audio_files, verbose=True):
        """Extract features for multiple audio files"""
        features_list = []
        total = len(audio_files)
        
        iterator = tqdm(audio_files, desc="Extracting features") if verbose else audio_files
        
        for audio_file in iterator:
            features = self.extract_features(audio_file)
            features_list.append(features)
        
        return pd.DataFrame(features_list)

# =========================
# DATA LOADER - RANDOM SAMPLING
# =========================
class AudioDataLoader:
    """Loads audio files from all paths with random sampling"""
    
    def __init__(self, audio_paths, max_samples=MAX_SAMPLES):
        self.audio_paths = audio_paths
        self.max_samples = max_samples
    
    def find_all_audio_files(self):
        """Find all FLAC/WAV files in the given paths"""
        all_files = []
        
        print("\n📂 Scanning for audio files...")
        for path in self.audio_paths:
            if not os.path.exists(path):
                print(f"   ⚠️ Path not found: {path}")
                continue
            
            # Find FLAC files
            flac_files = glob.glob(os.path.join(path, "*.flac"))
            wav_files = glob.glob(os.path.join(path, "*.wav"))
            
            files = flac_files + wav_files
            all_files.extend(files)
            print(f"   ✓ {os.path.basename(path)}: {len(files)} files")
        
        # Remove duplicates
        all_files = list(set(all_files))
        print(f"\n   Total unique audio files: {len(all_files)}")
        
        return all_files
    
    def load_random_sample(self):
        """Load random sample of audio files for training"""
        all_files = self.find_all_audio_files()
        
        if not all_files:
            print("❌ No audio files found!")
            return []
        
        # Random sample
        sample_size = min(self.max_samples, len(all_files))
        sampled_files = random.sample(all_files, sample_size)
        
        print(f"\n🎲 Randomly sampled {len(sampled_files)} files for training")
        
        return sampled_files

# =========================
# ANOMALY DETECTOR
# =========================
class AudioAnomalyDetector:
    """Detects fake/spoof audio using anomaly detection"""
    
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=EMBEDDING_DIM)
        self.model = None
        self.is_trained = False
        self.feature_names = None
    
    def train(self, audio_files, contamination=0.1):
        """
        Train on SPOOF/FAKE audio files
        Uses Isolation Forest for anomaly detection
        """
        print("\n" + "=" * 80)
        print("🏋️ TRAINING AUDIO ANOMALY DETECTOR")
        print("=" * 80)
        print(f"Training on {len(audio_files)} spoof audio files")
        
        # Extract features
        print("\n🔧 Extracting features...")
        X = self.feature_extractor.extract_batch(audio_files, verbose=True)
        
        self.feature_names = X.columns.tolist()
        print(f"   Features extracted: {X.shape[1]}")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        print("   Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimension for embedding
        print("   Generating embeddings...")
        X_embedding = self.pca.fit_transform(X_scaled)
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"   PCA explained variance: {explained_var:.2%}")
        
        # Train Isolation Forest (anomaly detection)
        print("\n🚀 Training Isolation Forest...")
        self.model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_SEED,
            n_estimators=200,
            max_samples='auto',
            n_jobs=-1
        )
        
        start_time = time.time()
        self.model.fit(X_scaled)
        training_time = time.time() - start_time
        
        # Evaluate on training data (for reference)
        predictions = self.model.predict(X_scaled)
        anomaly_count = np.sum(predictions == -1)
        
        print(f"\n📊 Training Results:")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Anomalies detected in training: {anomaly_count}/{len(X_scaled)} ({anomaly_count/len(X_scaled)*100:.1f}%)")
        print(f"   (Note: This is the contamination rate we set)")
        
        self.is_trained = True
        
        return {
            'training_samples': len(X_scaled),
            'features': X.shape[1],
            'anomaly_rate': anomaly_count/len(X_scaled),
            'training_time': training_time
        }
    
    def analyze_audio(self, audio_path):
        """
        Analyze audio file - returns risk score (0-1)
        Higher score = more likely to be FAKE/SPOOF
        """
        if not self.is_trained:
            return self._fallback_analysis(audio_path)
        
        if not os.path.exists(audio_path):
            return {
                'modality': 'audio',
                'risk_score': 0.5,
                'error': 'File not found',
                'audio_path': audio_path
            }
        
        # Extract features
        features = self.feature_extractor.extract_features(audio_path)
        X = pd.DataFrame([features])
        X = X.fillna(0)
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score
        # Isolation Forest: -1 = anomaly, 1 = normal
        # Convert to risk score (0-1, higher = more likely fake/spoof)
        anomaly_score = self.model.decision_function(X_scaled)[0]
        # Normalize to 0-1 range
        risk_score = 1 / (1 + np.exp(-anomaly_score))  # Sigmoid
        # Invert if needed (higher score = more anomalous)
        risk_score = 1 - risk_score
        
        # Generate embedding for fusion
        X_pca = self.pca.transform(X_scaled)
        embedding = X_pca[0] if len(X_pca) > 0 else np.zeros(EMBEDDING_DIM)
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = "🔴 CRITICAL"
            risk_text = "CRITICAL"
            recommendation = "⚠️ This audio shows strong signs of being SPOOFED/DEEPFAKE!"
        elif risk_score >= 0.6:
            risk_level = "🟠 HIGH"
            risk_text = "HIGH"
            recommendation = "⚠️ High risk - this audio likely contains synthetic/manipulated speech"
        elif risk_score >= 0.4:
            risk_level = "🟡 MEDIUM"
            risk_text = "MEDIUM"
            recommendation = "⚠️ Medium risk - unusual acoustic characteristics detected"
        elif risk_score >= 0.2:
            risk_level = "🔵 LOW"
            risk_text = "LOW"
            recommendation = "✅ Low risk - minor anomalies, likely legitimate"
        else:
            risk_level = "🟢 SAFE"
            risk_text = "SAFE"
            recommendation = "✅ This audio appears to be legitimate human speech"
        
        return {
            'modality': 'audio',
            'risk_score': risk_score,
            'safe_score': 1 - risk_score,
            'embedding': embedding,
            'confidence': abs(risk_score - 0.5) * 2,
            'risk_level': risk_level,
            'risk_text': risk_text,
            'is_phishing': risk_score > 0.5,
            'audio_path': audio_path,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'features': {
                'MFCC_Variance': np.std([features.get(f'mfcc_{i}_mean', 0) for i in range(10)]),
                'Spectral_Centroid': features.get('spec_centroid_mean', 0),
                'Zero_Crossing_Rate': features.get('zcr_mean', 0),
                'Spectral_Entropy': features.get('spectral_entropy', 0),
                'HNR': features.get('hnr', 0)
            }
        }
    
    def _fallback_analysis(self, audio_path):
        """Fallback when model not trained"""
        return {
            'modality': 'audio',
            'risk_score': 0.5,
            'safe_score': 0.5,
            'embedding': np.zeros(EMBEDDING_DIM),
            'confidence': 0,
            'risk_level': "⚠️ MODEL NOT TRAINED",
            'risk_text': "UNKNOWN",
            'is_phishing': False,
            'audio_path': audio_path,
            'recommendation': "Please train the model first",
            'features': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def save(self, path):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, 'model.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(path, 'pca.pkl'))
        joblib.dump(self.feature_names, os.path.join(path, 'feature_names.pkl'))
        print(f"\n✓ Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = joblib.load(os.path.join(path, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        self.pca = joblib.load(os.path.join(path, 'pca.pkl'))
        self.feature_names = joblib.load(os.path.join(path, 'feature_names.pkl'))
        self.is_trained = True
        print(f"✓ Model loaded from {path}")

# =========================
# INTERACTIVE TESTER
# =========================
class AudioTester:
    def __init__(self, detector):
        self.detector = detector
    
    def run(self):
        print("\n" + "=" * 100)
        print("🎯 INTERACTIVE AUDIO TESTER")
        print("=" * 100)
        print("\nCommands:")
        print("  • Enter audio file path to test")
        print("  • 'folder' - Test all audio files in a folder")
        print("  • 'sample' - Test random samples from your dataset")
        print("  • 'quit' - Exit")
        print("-" * 100)
        
        while True:
            print("\n" + "-" * 100)
            user_input = input("🎵 Enter audio file path: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'folder':
                self._test_folder()
            
            elif user_input.lower() == 'sample':
                self._test_samples()
            
            elif user_input:
                self._test_audio(user_input)
    
    def _test_audio(self, audio_path):
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return
        
        result = self.detector.analyze_audio(audio_path)
        
        print("\n" + "=" * 80)
        print("📊 AUDIO ANALYSIS RESULT")
        print("=" * 80)
        print(f"File: {result['audio_path']}")
        print(f"\nRisk Score: {result['risk_score']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Classification: {'⚠️ FAKE/SPOOF' if result['is_phishing'] else '✅ REAL/LEGITIMATE'}")
        
        bar_length = 40
        filled = int(bar_length * result['risk_score'])
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\nRisk Meter: [{bar}]")
        
        print(f"\n{result['recommendation']}")
        
        print("\n🔬 Fusion Output:")
        print(f"   Modality: {result['modality']}")
        print(f"   Embedding Shape: {len(result['embedding'])}-dim")
        print(f"   Embedding Preview: [{result['embedding'][0]:.4f}, {result['embedding'][1]:.4f}, ...]")
        print("=" * 80)
    
    def _test_folder(self):
        folder = input("📁 Enter folder path: ").strip()
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            return
        
        audio_files = []
        for ext in ['*.flac', '*.wav']:
            audio_files.extend(glob.glob(os.path.join(folder, ext)))
        
        if not audio_files:
            print(f"No audio files found in {folder}")
            return
        
        print(f"\n📊 Testing {len(audio_files)} files...")
        for audio_file in audio_files[:10]:  # Limit to 10
            result = self.detector.analyze_audio(audio_file)
            bar = '█' * int(20 * result['risk_score']) + '░' * (20 - int(20 * result['risk_score']))
            print(f"{result['risk_text']:<8} [{bar}] {result['risk_score']:.1%} | {os.path.basename(audio_file)}")
    
    def _test_samples(self):
        """Test random samples from your datasets"""
        # Find some audio files
        test_files = []
        for path in AUDIO_PATHS:
            if os.path.exists(path):
                files = glob.glob(os.path.join(path, "*.flac"))[:5]
                test_files.extend(files)
        
        if not test_files:
            print("No sample files found")
            return
        
        print(f"\n📊 Testing {len(test_files)} sample files:\n")
        for audio_file in test_files:
            result = self.detector.analyze_audio(audio_file)
            bar = '█' * int(30 * result['risk_score']) + '░' * (30 - int(30 * result['risk_score']))
            print(f"{result['risk_level']:<12} [{bar}] {result['risk_score']:.1%} | {os.path.basename(audio_file)}")

# =========================
# MAIN
# =========================
def main():
    print("\n🚀 Starting Audio Phishing Detection System")
    
    # Check for existing model
    if os.path.exists(os.path.join(MODEL_SAVE_PATH, 'model.pkl')):
        print(f"\n✅ Found existing model")
        response = input("Use existing model? (y/n): ").strip().lower()
        if response == 'y':
            detector = AudioAnomalyDetector()
            detector.load(MODEL_SAVE_PATH)
            tester = AudioTester(detector)
            tester.run()
            return
    
    # Load audio files
    loader = AudioDataLoader(AUDIO_PATHS, MAX_SAMPLES)
    audio_files = loader.load_random_sample()
    
    if not audio_files:
        print("\n❌ No audio files found! Please check your paths.")
        return
    
    # Train detector
    detector = AudioAnomalyDetector()
    results = detector.train(audio_files)
    
    # Save model
    detector.save(MODEL_SAVE_PATH)
    
    # Quick test
    print("\n" + "=" * 80)
    print("🎯 QUICK TEST")
    print("=" * 80)
    
    # Get a few test files
    test_files = audio_files[:5]
    print("\nTesting sample audio files:\n")
    for audio_file in test_files:
        result = detector.analyze_audio(audio_file)
        bar = '█' * int(30 * result['risk_score']) + '░' * (30 - int(30 * result['risk_score']))
        print(f"{result['risk_level']:<12} [{bar}] {result['risk_score']:.1%} | {os.path.basename(audio_file)}")
        print(f"   → {'⚠️ FAKE/SPOOF' if result['is_phishing'] else '✅ REAL/LEGITIMATE'}")
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("🎯 ENTERING INTERACTIVE MODE")
    print("=" * 80)
    
    tester = AudioTester(detector)
    tester.run()

if __name__ == "__main__":
    main()