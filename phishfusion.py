#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CORE FUSION ENGINE - Multi-Modal Phishing Detection System
- Fuses outputs from Text, URL, Image, Audio, and Video modules
- Implements Embedding Correlation (Cosine Similarity)
- Dynamic Weighting based on Confidence
- Prepares data for Gemini AI Explanation
"""

import numpy as np
import scipy.spatial.distance as dist
from datetime import datetime
import json

print("=" * 60)
print("[PHISHGUARD] CORE FUSION ENGINE - LOADING")
print("=" * 60)

class FusionEngine:
    def __init__(self):
        # Base importance weights for different modalities
        # URLs and Texts carry the highest standard weights in typical phishing
        self.base_weights = {
            'text': 1.2,
            'url': 1.5,
            'image': 1.0,
            'audio': 1.0,
            'video': 1.1
        }
        
    def _align_embeddings(self, emb1, emb2):
        """
        Ensure embeddings are the same dimension for Cosine Similarity.
        Text model outputs 768-dim, while Audio/Image output 64-dim.
        We truncate to the minimum overlapping topological space.
        """
        min_len = min(len(emb1), len(emb2))
        return emb1[:min_len], emb2[:min_len]

    def _compute_embedding_correlation(self, results_dict):
        """
        5.1 EMBEDDING CORRELATION (Core Architecture Requirement)
        Computes cosine similarity between all available modality embeddings.
        Returns an agreement multiplier.
        """
        keys = [k for k in results_dict.keys() if 'embedding' in results_dict[k]]
        if len(keys) < 2:
            return 1.0 # Neutral multiplier if only 1 modality is present
            
        similarities = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                emb1 = results_dict[keys[i]]['embedding']
                emb2 = results_dict[keys[j]]['embedding']
                
                # Align dimensions
                e1, e2 = self._align_embeddings(emb1, emb2)
                
                # Prevent math errors on zero vectors (fallback values)
                if np.linalg.norm(e1) == 0 or np.linalg.norm(e2) == 0:
                    continue
                    
                try:
                    # 1 - Cosine Distance = Cosine Similarity
                    sim = 1.0 - dist.cosine(e1, e2)
                    # Absolute correlation (we care about pattern alignment)
                    similarities.append(abs(sim))
                except Exception as e:
                    continue
                    
        if not similarities:
            return 1.0
            
        avg_similarity = np.mean(similarities)
        
        # 5.2 INTERPRETATION
        # High similarity (close to 1) -> models agree -> boost final risk confidence
        # Low similarity -> disagreement -> reduce final risk confidence
        # Maps [0, 1] similarity to [0.8, 1.2] confidence multiplier
        agreement_multiplier = 0.8 + (avg_similarity * 0.4) 
        
        return agreement_multiplier

    def fuse(self, multi_modal_inputs):
        """
        5.3 DYNAMIC WEIGHTING & FINAL SCORE
        Expects a dictionary like: { 'url': url_result, 'text': text_result, ... }
        """
        if not multi_modal_inputs:
            return {"error": "No input modalities provided"}

        total_weighted_risk = 0.0
        total_weight = 0.0
        
        individual_risks = {}
        flags = []
        
        # 1. INDEPENDENT DYNAMIC WEIGHTING
        for modality, result in multi_modal_inputs.items():
            if result is None or 'risk_score' not in result:
                continue
                
            risk = result['risk_score']
            # Confidence is derived from how polarizing the prediction was (e.g. 0.9 or 0.1 has high conf)
            conf = result.get('confidence', 1.0) 
            base_weight = self.base_weights.get(modality, 1.0)
            
            # Dynamic Weighting formula based on architecture doc
            adjusted_weight = base_weight * conf
            
            total_weighted_risk += (risk * adjusted_weight)
            total_weight += adjusted_weight
            
            individual_risks[modality] = float(risk)
            
            # Track major red flags
            if risk > 0.7:
                flags.append(f"High risk detected in {modality} component.")
            
        if total_weight == 0:
            return {"error": "Failed to calculate weights"}
            
        # 2. EMBEDDING CORRELATION SCORE
        agreement_multiplier = self._compute_embedding_correlation(multi_modal_inputs)
        
        # 3. FINAL SCORE CALCULATION
        base_fusion_score = total_weighted_risk / total_weight
        
        # Final Score = \sum (weight × risk × confidence) adjusted by correlation agreement
        final_risk_score = base_fusion_score * agreement_multiplier
        
        # Constrain to 0.01 - 0.99 for frontend display
        final_risk_score = min(max(final_risk_score, 0.01), 0.99)
        
        # 4. 5.5 OUTPUT CATEGORIZATION
        if final_risk_score >= 0.8:
            risk_level = "🔴 CRITICAL"
            action = "DO NOT OPEN OR INTERACT. IMMINENT THREAT."
        elif final_risk_score >= 0.6:
            risk_level = "🟠 HIGH"
            action = "VERY SUSPICIOUS. High probability of phishing."
        elif final_risk_score >= 0.4:
            risk_level = "🟡 MEDIUM"
            action = "PROCEED WITH CAUTION. Verify source."
        elif final_risk_score >= 0.2:
            risk_level = "🔵 LOW"
            action = "GENERALLY SAFE. Keep minor awareness."
        else:
            risk_level = "🟢 SAFE"
            action = "SAFE. No recognized threats."
            
        return {
            "final_risk_score": final_risk_score,
            "risk_level": risk_level,
            "action_required": action,
            "agreement_multiplier": float(agreement_multiplier),
            "modalities_analyzed": list(individual_risks.keys()),
            "individual_risks": individual_risks,
            "red_flags": flags,
            "timestamp": datetime.now().isoformat()
        }

# =========================
# SYSTEM TESTER ALGORITHM
# =========================
def test_fusion_engine():
    print("\n" + "=" * 80)
    print("🧪 RUNNING FUSION SIMULATION")
    print("=" * 80)
    
    engine = FusionEngine()
    
    # Simulating the exact outputs from the 4 models
    # Example 1: High Agreement Phishing Attack (URL + Text + Image all agree it's phishing)
    mock_high_agreement_phish = {
        'url': {
            'risk_score': 0.85, 'confidence': 0.9, 'embedding': np.random.rand(64)
        },
        'text': {
            'risk_score': 0.92, 'confidence': 0.95, 'embedding': np.random.rand(768) # Testing 768-dim truncation
        },
        'image': {
            'risk_score': 0.78, 'confidence': 0.8, 'embedding': np.random.rand(64)
        }
    }
    
    # We artificially make the embeddings line up deeply to simulate high correlation
    shared_vector = np.ones(64) * 0.8
    mock_high_agreement_phish['url']['embedding'][:64] = shared_vector
    mock_high_agreement_phish['text']['embedding'][:64] = shared_vector
    mock_high_agreement_phish['image']['embedding'][:64] = shared_vector

    print("\n--- TEST CASE 1: Coordinated Phishing Attack (URL + Text + Image) ---")
    result_high = engine.fuse(mock_high_agreement_phish)
    print(json.dumps(result_high, indent=4))
    
    
    # Example 2: Disagreement (URL looks safe, but Audio is deepfake)
    mock_disagreement = {
        'url': {
            'risk_score': 0.10, 'confidence': 0.8, 'embedding': np.random.rand(64)
        },
        'audio': {
            'risk_score': 0.88, 'confidence': 0.9, 'embedding': np.zeros(64) # Completely different vector
        }
    }
    
    print("\n--- TEST CASE 2: Vishing Attack (Safe URL + Deepfake Audio) ---")
    result_discord = engine.fuse(mock_disagreement)
    print(json.dumps(result_discord, indent=4))

if __name__ == "__main__":
    test_fusion_engine()
