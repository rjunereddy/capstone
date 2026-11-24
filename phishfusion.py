#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CORE FUSION ENGINE - Multi-Modal Phishing Detection System
- Fuses outputs from URL, Text, Image, Audio, and Video modules
- Implements 128-D Shared Latent Space Projection & Embedding Correlation
- Dynamic Confidence-Aware Weighting: W_i = B_i * C_i
- Self-contained implementation supporting both PyTorch and NumPy
"""

import os
import json
import numpy as np
from datetime import datetime

# Optional PyTorch support with seamless NumPy fallback
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

# Optional SciPy distance support with NumPy fallback
try:
    import scipy.spatial.distance as dist
    HAS_SCIPY = True
except ImportError:
    dist = None
    HAS_SCIPY = False


def _get_deterministic_weights(in_dim, out_dim, seed):
    """Generates a deterministic semi-orthogonal projection matrix (out_dim, in_dim)."""
    rng = np.random.RandomState(seed)
    max_dim = max(in_dim, out_dim)
    raw = rng.randn(max_dim, max_dim)
    q, _ = np.linalg.qr(raw)
    W = q[:out_dim, :in_dim].astype(np.float32)
    row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / np.where(row_norms == 0, 1.0, row_norms)
    return W


class SharedFusionSpace:
    """
    Shared Latent Space Projection Network (128-D)
    Projects heterogeneous embeddings into a unified metric space for cross-modal correlation:
      - Text:  768-D -> 128-D
      - URL:    64-D -> 128-D
      - Image:  64-D -> 128-D
      - Audio:  64-D -> 128-D
      - Video:  64-D -> 128-D
    """
    def __init__(self, fusion_dim=128):
        self.fusion_dim = fusion_dim
        
        # Pre-compute deterministic projection matrices
        self.weights = {
            'text':  _get_deterministic_weights(768, fusion_dim, seed=42),
            'url':   _get_deterministic_weights(64,  fusion_dim, seed=43),
            'image': _get_deterministic_weights(64,  fusion_dim, seed=44),
            'audio': _get_deterministic_weights(64,  fusion_dim, seed=45),
            'video': _get_deterministic_weights(64,  fusion_dim, seed=46),
        }
        self.biases = {
            k: np.zeros(fusion_dim, dtype=np.float32) for k in self.weights
        }

    def project(self, modality, embedding):
        """Projects a modality embedding into the shared 128-D space."""
        if embedding is None:
            return np.zeros(self.fusion_dim, dtype=np.float32)
            
        if HAS_TORCH and isinstance(embedding, torch.Tensor):
            x = embedding.detach().cpu().numpy().flatten()
        else:
            x = np.asarray(embedding, dtype=np.float32).flatten()
            
        W = self.weights.get(modality)
        if W is None:
            return np.zeros(self.fusion_dim, dtype=np.float32)
            
        in_dim = W.shape[1]
        if len(x) > in_dim:
            x = x[:in_dim]
        elif len(x) < in_dim:
            x = np.pad(x, (0, in_dim - len(x)))
            
        b = self.biases.get(modality, np.zeros(self.fusion_dim, dtype=np.float32))
        z = np.dot(W, x) + b
        
        # L2-normalize projection
        norm = np.linalg.norm(z)
        if norm > 1e-8:
            z = z / norm
        return z


class FusionEngine:
    def __init__(self):
        # Base importance weights for modalities (as per PhishGuard Capstone specification)
        self.base_weights = {
            'url':   1.5,
            'text':  1.2,
            'video': 1.1,
            'image': 1.0,
            'audio': 1.0
        }
        self.fusion_space = SharedFusionSpace(fusion_dim=128)

    def _cosine_similarity(self, e1, e2):
        """Computes cosine similarity between two projected vectors."""
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 == 0 or n2 == 0:
            return 0.0
            
        if HAS_SCIPY and dist is not None:
            try:
                return float(1.0 - dist.cosine(e1, e2))
            except Exception:
                pass
                
        # Pure NumPy fallback
        return float(np.dot(e1, e2) / (n1 * n2 + 1e-8))

    def _compute_embedding_correlation(self, results_dict):
        """
        Cross-Modal Correlation via Shared 128-D Latent Space.
        Computes pairwise cosine similarity between all available modality embeddings.
        Returns an agreement multiplier in [0.8, 1.2].
        """
        keys = [k for k in results_dict.keys() if 'embedding' in results_dict[k] and results_dict[k]['embedding'] is not None]
        if len(keys) < 2:
            return 1.0  # Neutral multiplier if only 1 modality is available
            
        # Project all available embeddings into the shared space
        projected = {}
        for k in keys:
            projected[k] = self.fusion_space.project(k, results_dict[k]['embedding'])
            
        similarities = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                sim = self._cosine_similarity(projected[k1], projected[k2])
                similarities.append(sim)
                
        if not similarities:
            return 1.0
            
        avg_similarity = float(np.mean(similarities))
        
        # Map average similarity [-1, 1] to agreement multiplier [0.8, 1.2]
        # Multiplier = 1.0 + 0.2 * S_avg
        agreement_multiplier = 1.0 + (avg_similarity * 0.2)
        agreement_multiplier = max(0.8, min(1.2, agreement_multiplier))
        return agreement_multiplier

    def fuse(self, multi_modal_inputs):
        """
        Dynamic Input-Adaptive Confidence-Aware Fusion.
        Aggregates available modality risks:
          R_base = sum(W_i * R_i) / sum(W_i) where W_i = B_i * C_i
          R_final = clip(R_base * alpha, 0.01, 0.99)
        """
        if not multi_modal_inputs:
            return {"error": "No input modalities provided"}

        total_weighted_risk = 0.0
        total_weight = 0.0
        
        individual_risks = {}
        flags = []
        
        # 1. Independent Dynamic Confidence-Aware Weighting
        for modality, result in multi_modal_inputs.items():
            if result is None or 'risk_score' not in result:
                continue
                
            risk = float(result['risk_score'])
            conf = float(result.get('confidence', 0.85))
            base_weight = float(self.base_weights.get(modality, 1.0))
            
            # Confidence-adjusted weight: W_i = B_i * C_i
            adjusted_weight = base_weight * max(conf, 0.1)
            
            total_weighted_risk += (risk * adjusted_weight)
            total_weight += adjusted_weight
            
            individual_risks[modality] = round(risk, 4)
            
            # Extract indicators
            indicators = result.get('indicators') or []
            for ind in indicators:
                if ind and ind not in flags:
                    flags.append(ind)
                    
            if risk >= 0.70 and not any(modality in f.lower() for f in flags):
                flags.append(f"High risk signal detected in {modality.upper()} component")

        if total_weight == 0:
            return {"error": "Failed to calculate weights from available modalities"}
            
        # 2. Embedding Correlation Multiplier via Shared Latent Space
        agreement_multiplier = self._compute_embedding_correlation(multi_modal_inputs)
        
        # 3. Base & Final Fusion Score
        base_fusion_score = total_weighted_risk / total_weight
        
        # Asymmetric threat preservation:
        # If any specialized detector identifies high/critical threat (>= 0.65),
        # prevent naive arithmetic dilution by benign modalities (core paper contribution)
        max_threat = max(individual_risks.values()) if individual_risks else 0.0
        if max_threat >= 0.65:
            base_fusion_score = max(base_fusion_score, max_threat * 0.88)
            
        final_risk_score = base_fusion_score * agreement_multiplier
        final_risk_score = float(np.clip(final_risk_score, 0.01, 0.99))
        
        # 4. Threat Level Classification
        if final_risk_score >= 0.80:
            risk_level = "🔴 CRITICAL"
            action = "COORDINATED PHISHING ATTACK DETECTED. DO NOT INTERACT OR ENTER SENSITIVE DATA."
        elif final_risk_score >= 0.60:
            risk_level = "🟠 HIGH"
            action = "VERY SUSPICIOUS WEBSITE. Strong phishing signals present. Avoid entering credentials."
        elif final_risk_score >= 0.40:
            risk_level = "🟡 MEDIUM"
            action = "PROCEED WITH CAUTION. Potential security anomalies detected. Verify domain authenticity."
        elif final_risk_score >= 0.20:
            risk_level = "🔵 LOW"
            action = "MINIMAL RISK DETECTED. Generally safe, but remain observant."
        else:
            risk_level = "🟢 SAFE"
            action = "VERIFIED SAFE. No recognized phishing or deepfake threats detected."
            
        return {
            "final_risk_score": round(final_risk_score, 4),
            "risk_level": risk_level,
            "action_required": action,
            "agreement_multiplier": round(float(agreement_multiplier), 4),
            "modalities_analyzed": list(individual_risks.keys()),
            "individual_risks": individual_risks,
            "red_flags": flags[:6],
            "timestamp": datetime.now().isoformat()
        }


def test_fusion_engine():
    print("=" * 70)
    print("🧪 PHISHGUARD DYNAMIC FUSION ENGINE — SELF-TEST")
    print("=" * 70)
    
    engine = FusionEngine()
    
    # Test Case 1: High agreement phishing attack (URL + Text + Image)
    test_1 = {
        'url':   {'risk_score': 0.88, 'confidence': 0.92, 'embedding': np.ones(64) * 0.5, 'indicators': ['High-risk TLD', 'Brand impersonation']},
        'text':  {'risk_score': 0.91, 'confidence': 0.95, 'embedding': np.ones(768) * 0.5, 'indicators': ['Urgency manipulation']},
        'image': {'risk_score': 0.82, 'confidence': 0.85, 'embedding': np.ones(64) * 0.5, 'indicators': ['Cloned login layout']}
    }
    r1 = engine.fuse(test_1)
    print("Scenario 1 (Coordinated Phishing Attack):")
    print(json.dumps(r1, indent=2))
    
    # Test Case 2: Vishing attack (Safe URL + Malicious Audio)
    test_2 = {
        'url':   {'risk_score': 0.05, 'confidence': 0.95, 'embedding': np.zeros(64), 'indicators': ['Known legitimate domain']},
        'audio': {'risk_score': 0.85, 'confidence': 0.90, 'embedding': np.random.randn(64) * 0.3, 'indicators': ['Synthetic voice OTP lure']}
    }
    r2 = engine.fuse(test_2)
    print("\nScenario 2 (Vishing Lure with Conflicting Evidence):")
    print(json.dumps(r2, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    test_fusion_engine()
