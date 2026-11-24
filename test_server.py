#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD COMPREHENSIVE END-TO-END TEST SUITE
================================================
Verifies all 5 modalities, individual modular endpoints, Dynamic Fusion Engine,
SHAP explainability, and the full multi-modal detection pipeline.
Supports both direct Flask test_client (in-process) and live HTTP server testing.
"""

import json
import time
import sys

# Try connecting to live server first; fall back to in-process test client
LIVE_URL = 'http://127.0.0.1:5000'
use_live = False

try:
    import urllib.request
    urllib.request.urlopen(f"{LIVE_URL}/health", timeout=1.0)
    use_live = True
except Exception:
    use_live = False

if not use_live:
    from server import app
    client = app.test_client()

def get(path):
    if use_live:
        req = urllib.request.Request(f"{LIVE_URL}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode('utf-8'))
    else:
        resp = client.get(path)
        return json.loads(resp.data.decode('utf-8'))

def post(path, payload):
    if use_live:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{LIVE_URL}{path}",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode('utf-8'))
    else:
        resp = client.post(path, data=json.dumps(payload), content_type='application/json')
        return json.loads(resp.data.decode('utf-8'))

print("=" * 80)
print(f"🛡️  PHISHGUARD AI: COMPREHENSIVE END-TO-END VERIFICATION")
print(f"   Execution Mode: {'Live HTTP Server (' + LIVE_URL + ')' if use_live else 'In-Process Flask Engine'}")
print("=" * 80)

passed = 0
total = 0

def check(condition, message, detail=""):
    global passed, total
    total += 1
    tag = "✅ PASS" if condition else "❌ FAIL"
    if condition:
        passed += 1
    print(f"[{tag}] {message} {detail}")
    return condition

# ── 1. HEALTH & METRICS CHECK ────────────────────────────────────────────────
print("\n[Phase 1] System Health & Model Metadata Inspection...")
health = get('/health')
check(health.get('status') == 'online', "System status is online")
check('url' in health.get('models', {}), "URL detector registered", f"({health['models']['url'][:30]}...)")
check('text' in health.get('models', {}), "Text detector registered", f"({health['models']['text'][:30]}...)")
check('image' in health.get('models', {}), "Image detector registered", f"({health['models']['image'][:30]}...)")
check('audio' in health.get('models', {}), "Audio detector registered", f"({health['models']['audio'][:30]}...)")
check('video' in health.get('models', {}), "Video detector registered", f"({health['models']['video'][:30]}...)")
check(health.get('metrics', {}).get('url_accuracy') > 0.95, "URL Model Accuracy reported >= 95%")
check(health.get('metrics', {}).get('fusion_auc') > 0.98, "Dynamic Fusion AUC reported >= 98%")

# ── 2. INDIVIDUAL MODALITY ENDPOINTS ─────────────────────────────────────────
print("\n[Phase 2] Individual Modular Endpoints Verification...")
u_res = post('/analyze/url', {'url': 'http://paypal-security-update.tk/login'})
check(u_res.get('modality') == 'url' and u_res.get('risk_score', 0) > 0.60, 
      "POST /analyze/url flags suspicious lookalike domain", f"Risk: {u_res.get('risk_score', 0):.1%}")

t_res = post('/analyze/text', {'text': 'URGENT: Your bank account will be suspended within 24 hours. Enter your password and OTP immediately.'})
check(t_res.get('modality') == 'text' and t_res.get('risk_score', 0) > 0.60, 
      "POST /analyze/text detects urgent credential harvesting", f"Risk: {t_res.get('risk_score', 0):.1%}")

i_res = post('/analyze/image', {'url': 'http://paypal-login-portal.tk', 'dom_features': {'has_password': True, 'num_images': 4}})
check(i_res.get('modality') == 'image' and i_res.get('risk_score', 0) > 0.50, 
      "POST /analyze/image identifies cloned login layout", f"Risk: {i_res.get('risk_score', 0):.1%}")

a_res = post('/analyze/audio', {'url': 'http://bank-support-call.xyz', 'text': 'Voice automated verification for your OTP', 'dom_features': {'has_audio': True}})
check(a_res.get('modality') == 'audio' and a_res.get('risk_score', 0) > 0.50, 
      "POST /analyze/audio detects synthetic voice vishing lure", f"Risk: {a_res.get('risk_score', 0):.1%}")

v_res = post('/analyze/video', {'url': 'http://ceo-urgent-broadcast.biz', 'text': 'Important message from company leadership', 'dom_features': {'has_video': True}})
check(v_res.get('modality') == 'video' and v_res.get('risk_score', 0) > 0.50, 
      "POST /analyze/video flags deepfake video artifact inconsistency", f"Risk: {v_res.get('risk_score', 0):.1%}")

# ── 3. FULL MULTI-MODAL PIPELINE SCENARIOS ────────────────────────────────────
print("\n[Phase 3] Full Multi-Modal Pipeline Scenarios...")

scenarios = [
    # 1. Benign known domains
    {
        "name": "Legitimate Search Engine (Google)",
        "payload": {"url": "https://www.google.com", "text": "Search the world's information, including webpages, images, videos and more."},
        "expect_high": False, "max_score": 0.25
    },
    {
        "name": "Authentic Google Login Portal (Accounts Google)",
        "payload": {
            "url": "https://accounts.google.com/v3/signin/identifier?authuser=0&continue=https://mail.google.com",
            "title": "Sign in - Google Accounts",
            "text": "Sign in to continue to Gmail. Enter your email or phone number. Next. Forgot email?",
            "dom_features": {"has_password": True, "num_images": 2}
        },
        "expect_high": False, "max_score": 0.20
    },
    {
        "name": "Authentic Gmail Application (Mail Google)",
        "payload": {
            "url": "https://mail.google.com/mail/u/1/#inbox",
            "title": "Inbox (12) - user@gmail.com - Gmail",
            "text": "Primary Promotions Social. Starred Snoozed Sent Drafts Meet New meeting.",
            "dom_features": {"num_images": 5}
        },
        "expect_high": False, "max_score": 0.20
    },
    {
        "name": "Legitimate Banking Portal (HDFC Bank)",
        "payload": {"url": "https://www.hdfcbank.com/personal/ways-to-bank/online-banking", "text": "Welcome to HDFC Bank NetBanking. Log in to your personal account securely."},
        "expect_high": False, "max_score": 0.30
    },
    {
        "name": "YouTube Search Results (Deepfake Query Catalog)",
        "payload": {
            "url": "https://www.youtube.com/results?search_query=tom+cruiess+deepfake",
            "title": "tom cruiess deepfake - YouTube",
            "text": "Filters Search results for tom cruiess deepfake. Watch latest uploads, shorts, and channels.",
            "dom_features": {"has_video": True}
        },
        "expect_high": False, "max_score": 0.20
    },
    {
        "name": "Legitimate Video Platform (Genuine Video)",
        "payload": {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
            "text": "Official music video with standard streaming playback.",
            "dom_features": {"has_video": True}
        },
        "expect_high": False, "max_score": 0.25
    },
    # 2. Deepfake on Video Platform
    {
        "name": "YouTube AI Deepfake Video (Celeb-DF Model Detection)",
        "payload": {
            "url": "https://www.youtube.com/watch?v=1yiOVUbsPcM",
            "title": "Very realistic Tom Cruise Deepfake | AI Tom Cruise",
            "text": "Very realistic Tom Cruise Deepfake | AI Tom Cruise. Watch this amazing deepfake created using artificial intelligence and faceswap technology.",
            "dom_features": {"has_video": True, "num_images": 4}
        },
        "expect_high": True, "min_score": 0.65
    },
    # 2. Phishing attacks
    {
        "name": "Credential Harvesting (PayPal Clone)",
        "payload": {
            "url": "http://paypal-security-check.tk/login/verify.php",
            "text": "URGENT: Your account has been locked. Verify your password and card details immediately to unlock.",
            "dom_features": {"has_password": True, "num_images": 3}
        },
        "expect_high": True, "min_score": 0.70
    },
    {
        "name": "AI Voice Scam (Vishing Lure)",
        "payload": {
            "url": "http://secure-voice-otp-verify-82.xyz",
            "text": "Automated voice alert: Suspicious debit detected. Call this number or enter your 6-digit OTP code.",
            "dom_features": {"has_audio": True}
        },
        "expect_high": True, "min_score": 0.60
    },
    {
        "name": "Deepfake Executive Video Attack",
        "payload": {
            "url": "http://urgent-video-login-75.net/exec-message",
            "text": "Special video broadcast from CEO: Immediate wire authorization required for confidential acquisition.",
            "dom_features": {"has_video": True, "has_password": True}
        },
        "expect_high": True, "min_score": 0.60
    },
    {
        "name": "Coordinated 5-Modality Critical Attack",
        "payload": {
            "url": "http://sbi-card-kyc-verify-urgent.tk/banking",
            "text": "Dear Customer, your SBI account is suspended due to pending KYC. Act now or permanent legal block occurs.",
            "dom_features": {"has_password": True, "num_images": 4, "has_audio": True, "has_video": True}
        },
        "expect_high": True, "min_score": 0.80
    }
]

for sc in scenarios:
    res = post('/analyze', sc['payload'])
    score = res.get('final_risk_score', 0)
    level = res.get('risk_level', '')
    modalities = res.get('modalities_analyzed', [])
    agreement = res.get('agreement_multiplier', 1.0)
    
    if sc['expect_high']:
        ok = score >= sc['min_score']
        cond_text = f"Score {score:.1%} >= {sc['min_score']:.1%} [{level}]"
    else:
        ok = score <= sc['max_score']
        cond_text = f"Score {score:.1%} <= {sc['max_score']:.1%} [{level}]"
        
    check(ok, sc['name'], f"-> {cond_text} | Modalities: {modalities} | Agreement: {agreement:.3f}x")

# ── 4. SHAP EXPLAINABILITY & XAI INSPECTION ──────────────────────────────────
print("\n[Phase 4] Explainable AI (XAI) & SHAP Inspection...")
sample_res = post('/analyze', scenarios[-1]['payload'])
shap = sample_res.get('shap_values', [])
ai_expl = sample_res.get('ai_explanation', '')

check(len(shap) > 0, "SHAP directional impacts generated", f"({len(shap)} factors: {[s['feature'] for s in shap[:2]]})")
check(len(ai_expl) > 40, "Contextual AI reasoning generated", f"({ai_expl[:65]}...)")

# ── 5. HISTORY TRACKING ──────────────────────────────────────────────────────
print("\n[Phase 5] Scan History Verification...")
hist = get('/history')
check(isinstance(hist, list) and len(hist) > 0, "Recent scans logged to history table", f"({len(hist)} records)")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"VERIFICATION COMPLETE: {passed}/{total} Checks Passed ({passed/total*100:.1f}%)")
if passed == total:
    print("🎉 ALL SYSTEMS FULLY OPERATIONAL — READY FOR UNIVERSITY CAPSTONE DEMO")
else:
    print(f"⚠️  {total - passed} checks failed. Inspect logs above.")
print("=" * 80)
