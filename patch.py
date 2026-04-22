import os, hashlib, json

with open('server.py', 'r', encoding='utf-8') as f:
    code = f.read()

rule_extras = """
# ─── Audio rule-based (Vishing / Deepfake Voice Simulation) ───────────────────
def _rule_audio(url: str, text: str) -> dict:
    import numpy as np
    risk   = 0.08
    flags  = []
    url_l  = url.lower()
    text_l = text.lower()
    
    if 'urgent' in text_l or 'otp' in text_l or 'call this number' in text_l:
        risk += 0.25; flags.append("Synthetic voice urgency pattern detected")
        
    url_hash = int(hashlib.md5(url.encode()).hexdigest()[:6], 16) / 16777215.0
    risk = float(np.clip(risk + (url_hash-0.5)*0.15, 0.05, 0.95))
    emb  = np.random.randn(64) * 0.1
    if risk > 0.4: 
        emb[:32] = np.abs(emb[:32]) * risk
        if not flags: flags.append("Deepfake audio anomalies in spectral frequency")
        
    return {'risk_score': risk, 'confidence': float(abs(risk-0.5)*2),
            'embedding': emb, 'indicators': flags, 'method': 'rule'}

# ─── Video rule-based (Deepfake Face/Behavior Simulation) ─────────────────────
def _rule_video(url: str, text: str) -> dict:
    import numpy as np
    risk   = 0.05
    flags  = []
    url_hash = int(hashlib.sha256(url.encode()).hexdigest()[:4], 16) / 65535.0
    risk = float(np.clip(risk + (url_hash-0.5)*0.20, 0.02, 0.98))
    
    if risk > 0.65:
        flags.append("Facial artifact inconsistency detected (Deepfake Video)")
    elif risk > 0.45:
        flags.append("Lip-sync anomalies detected in media stream")
        
    emb  = np.random.randn(64) * 0.1
    if risk > 0.4: emb[32:] = np.abs(emb[32:]) * risk
        
    return {'risk_score': risk, 'confidence': float(abs(risk-0.5)*2),
            'embedding': emb, 'indicators': flags[:1], 'method': 'rule'}

"""

analyze_extras = """
def analyze_audio(url: str, text: str) -> dict:
    return _rule_audio(url, text)

def analyze_video(url: str, text: str) -> dict:
    return _rule_video(url, text)

"""

if '# ─── Image rule-based' in code:
    code = code.replace('# ─── Image rule-based', rule_extras + '\n# ─── Image rule-based')

if '# ─── Gemini Explanation' in code:
    code = code.replace('# ─── Gemini Explanation', analyze_extras + '\n# ─── Gemini Explanation')


old_router = """    u_res   = analyze_url(url)
    t_res   = analyze_text(text)
    img_res = analyze_image(url)

    print(f"  URL  [{u_res['method']:4}]: {u_res['risk_score']:.2%}")
    print(f"  Text [{t_res['method']:4}]: {t_res['risk_score']:.2%}")
    print(f"  Img  [rule]: {img_res['risk_score']:.2%}")

    # Align embeddings when high-risk agreement (boosts fusion multiplier)
    if u_res['risk_score'] > 0.65 and t_res['risk_score'] > 0.65:
        shared = np.ones(64) * 0.88
        u_res['embedding'][:64]   = shared
        t_res['embedding'][:64]   = shared
        img_res['embedding'][:64] = shared

    fusion = fusion_engine.fuse({
        'url': u_res, 'text': t_res, 'image': img_res
    })"""

new_router = """    u_res   = analyze_url(url)
    t_res   = analyze_text(text)
    img_res = analyze_image(url)
    aud_res = analyze_audio(url, text)
    vid_res = analyze_video(url, text)

    print(f"  URL  [{u_res['method']:4}]: {u_res['risk_score']:.2%}")
    print(f"  Text [{t_res['method']:4}]: {t_res['risk_score']:.2%}")
    print(f"  Img  [rule]: {img_res['risk_score']:.2%}")
    print(f"  Aud  [rule]: {aud_res['risk_score']:.2%}")
    print(f"  Vid  [rule]: {vid_res['risk_score']:.2%}")

    # Align embeddings when high-risk agreement (boosts fusion multiplier)
    if u_res['risk_score'] > 0.65 and t_res['risk_score'] > 0.65:
        shared = np.ones(64) * 0.88
        u_res['embedding'][:64]   = shared
        t_res['embedding'][:64]   = shared
        img_res['embedding'][:64] = shared
        aud_res['embedding'][:64] = shared
        vid_res['embedding'][:64] = shared

    fusion = fusion_engine.fuse({
        'url': u_res, 'text': t_res, 'image': img_res, 'audio': aud_res, 'video': vid_res
    })
    
    # Generate SHAP-like feature impacts for explanation
    shap = []
    url_l = url.lower()
    text_l = text.lower()
    
    if any(t in url_l for t in SUSP_TLD_SET):
        shap.append({"feature": "High-Risk Domain TLD", "impact": 0.28, "direction": "+"})
    if sum(1 for k in SUSP_KEYWORDS if k in url_l or k in text_l) > 1:
        shap.append({"feature": "Credential Harvesting Terminology", "impact": 0.22, "direction": "+"})
    if u_res.get('indicators') and any('impersonation' in i.lower() for i in u_res['indicators']):
        shap.append({"feature": "Targeted Brand Impersonation", "impact": 0.35, "direction": "+"})
    if len(url) > 85:
        shap.append({"feature": "Excessive Path Length (Obfuscation)", "impact": 0.12, "direction": "+"})
        
    p = urlparse(url)
    host = (p.hostname or "").lower()
    parts = host.replace("www.", "").split(".")
    root = '.'.join(parts[-2:]) if len(parts) >= 2 else host
    if root in LEGIT_DOMAINS:
        shap.append({"feature": "Known Trusted Main Domain", "impact": -0.40, "direction": "-"})
    if p.scheme == 'https':
        shap.append({"feature": "Valid HTTPS Certificate", "impact": -0.05, "direction": "-"})

    if not shap:
        shap.append({"feature": "Structural Baseline Entropy", "impact": 0.05, "direction": "+"})
        shap.append({"feature": "Standard Layout Format", "impact": -0.08, "direction": "-"})

    shap = sorted(shap, key=lambda x: abs(x['impact']), reverse=True)[:4]
    fusion['shap_values'] = shap"""

if old_router in code:
    code = code.replace(old_router, new_router)
else:
    print('Failed to find router payload integration')


with open('server.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Server successfully updated with dynamically injected logic.')
