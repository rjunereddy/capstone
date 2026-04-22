#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD - MODEL TRAINING PIPELINE
=====================================
Trains all ML models for the Multi-Modal Phishing Detection System.

Models trained:
  1. URL Phishing Detector    - Ensemble (RF + GB + LR) on 57 structural features
  2. Text Phishing Classifier - TF-IDF (15k features) + Logistic Regression

Usage:
  python train_models.py          # Train all models
  python train_models.py --url    # URL model only
  python train_models.py --text   # Text model only
"""

import os, sys, re, json, random, string, time, math, argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from urllib.parse import urlparse, parse_qs
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
URL_MODEL_DIR  = os.path.join(MODELS_DIR, 'url_model')
TEXT_MODEL_DIR = os.path.join(MODELS_DIR, 'text_model')

for d in [MODELS_DIR, URL_MODEL_DIR, TEXT_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

URL_PHISHING_COUNT    = 25_000
URL_LEGITIMATE_COUNT  = 20_000
TEXT_PHISHING_COUNT   = 3_000
TEXT_LEGIT_COUNT      = 3_000
EMBEDDING_DIM         = 64

# ─── BANNER ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("  PHISHGUARD - MODEL TRAINING PIPELINE")
print("=" * 70)
print(f"  Output directory: {MODELS_DIR}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1:  URL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# ─── URL Training Data Vocabulary ────────────────────────────────────────────
LEGIT_DOMAINS = [
    "google.com","gmail.com","youtube.com","facebook.com","twitter.com",
    "linkedin.com","github.com","stackoverflow.com","wikipedia.org","amazon.com",
    "microsoft.com","apple.com","netflix.com","instagram.com","reddit.com",
    "discord.com","spotify.com","zoom.us","adobe.com","dropbox.com",
    "stripe.com","shopify.com","twitch.tv","medium.com","notion.so",
    "cloudflare.com","vercel.com","heroku.com","digitalocean.com","aws.amazon.com",
    # Indian
    "flipkart.com","amazon.in","myntra.com","bigbasket.com","swiggy.com",
    "zomato.com","paytm.com","phonepe.com","razorpay.com","freshworks.com",
    "zoho.com","irctc.co.in","sbi.co.in","hdfcbank.com","icicibank.com",
    "axisbank.com","india.gov.in","ndtv.com","thehindu.com","makemytrip.com",
    "bookmyshow.com","naukri.com","ola.com","oyo.com","meesho.com",
    "iitb.ac.in","iitd.ac.in","iitm.ac.in","iisc.ac.in","nptel.ac.in",
    "byjus.com","unacademy.com","vedantu.com","zerodha.com","groww.in",
    "policybazaar.com","practo.com","1mg.com","netmeds.com","pharmeasy.in",
]

BRAND_NAMES = [
    "paypal","google","amazon","facebook","apple","microsoft","netflix",
    "sbi","hdfc","icici","irctc","paytm","flipkart","whatsapp","instagram",
    "twitter","linkedin","ebay","yahoo","outlook","onedrive","dropbox",
    "zoom","spotify","ola","zomato","swiggy","razorpay","phonepe",
    "airtel","vodafone","jio","bsnl","lic","bajaj","tatamotors","reliance",
]

SUSPICIOUS_WORDS = [
    "verify","secure","login","signin","account","update","confirm",
    "banking","payment","credential","recovery","auth","validate",
    "required","urgent","alert","suspended","restricted","unlock",
    "activate","billing","invoice","refund","claim","prize","reward",
    "password","otp","kyc","aadhar","pancard","ifsc","swift",
]

SUSP_TLDS = [
    "tk","ml","ga","cf","xyz","club","online","site","website","space",
    "top","click","download","review","work","loan","win","bid","gq",
    "pw","cc","buzz","rest","link","life","live","shop","store",
]

LEGIT_TLDS = ["com","org","net","in","co.in","edu","gov","gov.in","co.uk","io"]

LEGIT_PATHS = [
    "/","/home","/about","/contact","/products","/services","/search",
    "/news","/blog","/help","/support","/faq","/shop","/cart","/checkout",
    "/account","/login","/docs","/api","/portfolio","/pricing","/team",
    "/privacy","/terms","/sitemap","/feed","/rss","/gallery","/events",
]

def _rand_str(min_len=4, max_len=12):
    n = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_lowercase, k=n))

def _rand_alnum(min_len=6, max_len=16):
    n = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _rand_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

def _rand_params(n=None):
    if n is None:
        n = random.randint(2, 7)
    keys = ["id","return","redirect","url","user","token","session","ref","key","auth","next","dest"]
    return "&".join(f"{random.choice(keys)}={_rand_alnum()}" for _ in range(n))

def _rand_hex_str(length=24):
    return ''.join(random.choices('0123456789abcdef', k=length))

# ─── Phishing URL Generator ───────────────────────────────────────────────────
def _gen_phishing_url():
    brand  = random.choice(BRAND_NAMES)
    word   = random.choice(SUSPICIOUS_WORDS)
    tld    = random.choice(SUSP_TLDS)
    cap    = random.choice(BRAND_NAMES)

    strategies = [
        # 1. Brand + suspicious TLD  e.g. paypal-verify.tk/login
        lambda: f"http://{brand}-{word}.{tld}/{random.choice(['login','verify','update','secure'])}",

        # 2. Brand in subdomain of suspicious domain
        lambda: f"http://{brand}.{_rand_str()}.{tld}/{word}?id={_rand_alnum()}",

        # 3. IP address URL
        lambda: f"http://{_rand_ip()}/{brand}/{word}?token={_rand_alnum()}&redirect=1",

        # 4. Brand impersonation in path of suspicious domain
        lambda: f"http://{_rand_str()}-{_rand_str()}.{tld}/{brand}/{word}/?{_rand_params()}",

        # 5. Long URL with many parameters
        lambda: f"http://{_rand_str()}.{random.choice(['com','net','org'])}/{_rand_str()}/{word}/{brand}?{_rand_params(6)}",

        # 6. Deep subdomain chain (paypal.secure.login.accounts.evil.tk)
        lambda: f"http://{brand}.{word}.{_rand_str()}.{_rand_str()}.{tld}/{word}",

        # 7. Hex/random domain (randomised domain names look machine-generated)
        lambda: f"http://{_rand_hex_str(14)}.{tld}/login?user={_rand_alnum()}&pass={_rand_alnum()}",

        # 8. URL with @ symbol (redirect trick)
        lambda: f"http://google.com@{_rand_str()}-{word}.{tld}/{brand}",

        # 9. Typosquatting: replace letters
        lambda: f"http://{brand.replace('a','@').replace('o','0').replace('i','1')}-secure.{random.choice(['com','net'])}/{word}",

        # 10. Brand + country + verify (common in Indian phishing)
        lambda: f"http://{brand}-india-{word}.{tld}/{word}?{_rand_params()}",

        # 11. Shortened URL simulation
        lambda: f"http://bit.ly/{_rand_alnum(6,10)}",

        # 12. Double slash in path (obfuscation)
        lambda: f"http://{_rand_str()}.{tld}/{word}//{brand}//login",

        # 13. Fake HTTPS (http but with 'secure' in domain)
        lambda: f"http://secure-{brand}-{random.randint(100,9999)}.{tld}/{word}",

        # 14. Very long path with random hex (common in phishing kits)
        lambda: f"http://{_rand_str()}.{tld}/{_rand_hex_str(32)}/{brand}_{word}.php",
    ]

    return random.choice(strategies)()


# ─── Legitimate URL Generator ─────────────────────────────────────────────────
def _gen_legitimate_url():
    domain = random.choice(LEGIT_DOMAINS)
    use_www = random.random() > 0.4
    prefix = "www." if use_www else ""
    path = random.choice(LEGIT_PATHS)

    # Sometimes add a simple query param
    query = ""
    if random.random() > 0.7:
        keys = ["q","page","lang","category","id","ref","tab","section"]
        query = f"?{random.choice(keys)}={_rand_str(3,8)}"

    return f"https://{prefix}{domain}{path}{query}"


def build_url_dataset(phishing_count=URL_PHISHING_COUNT, legit_count=URL_LEGITIMATE_COUNT):
    print(f"\n[1/6] Generating URL training data ({phishing_count} phishing + {legit_count} legitimate)...")
    t0 = time.time()

    phishing_urls  = [_gen_phishing_url()  for _ in range(phishing_count)]
    legit_urls     = [_gen_legitimate_url() for _ in range(legit_count)]

    urls   = phishing_urls + legit_urls
    labels = [1] * phishing_count + [0] * legit_count

    # Shuffle
    combined = list(zip(urls, labels))
    random.shuffle(combined)
    urls, labels = zip(*combined)

    print(f"     Done in {time.time()-t0:.1f}s  |  Total: {len(urls):,} URLs")
    return list(urls), list(labels)


# ─── URL Feature Extractor ────────────────────────────────────────────────────
SUSP_TLD_SET = {'.tk','.ml','.ga','.cf','.xyz','.club','.online','.site',
               '.website','.space','.top','.click','.download','.review',
               '.work','.loan','.win','.bid','.gq','.pw','.cc','.buzz',
               '.rest','.link','.life','.live','.shop','.store'}

SUSP_KEYWORD_SET = set(SUSPICIOUS_WORDS)

LEGIT_DOMAIN_SET = set(LEGIT_DOMAINS)

BRAND_SET = set(BRAND_NAMES)

SHORTENERS = {'bit.ly','tinyurl','goo.gl','ow.ly','cutt.ly','rb.gy','t.co',
              'tiny.cc','is.gd','shorturl','buff.ly','tr.im'}

SUSP_PARAMS = {'redirect','url','return','next','dest','goto','callback','ref'}


def extract_url_features(url: str) -> dict:
    """Extract 57 structural features from a URL."""
    f = {}
    try:
        p    = urlparse(url)
        host = (p.hostname or "").lower()
        path = p.path or ""
        qstr = p.query or ""
        url_l = url.lower()

        # 1-7: Lengths & ratios
        f['url_length']        = len(url)
        f['host_length']       = len(host)
        f['path_length']       = len(path)
        f['query_length']      = len(qstr)
        f['host_ratio']        = len(host) / max(len(url), 1)
        f['path_ratio']        = len(path) / max(len(url), 1)
        f['is_very_long']      = 1 if len(url) > 100 else 0

        # 8-22: Character counts
        f['num_dots']          = url.count('.')
        f['num_hyphens']       = url.count('-')
        f['num_underscores']   = url.count('_')
        f['num_slashes']       = url.count('/')
        f['num_qs']            = url.count('?')
        f['num_equals']        = url.count('=')
        f['num_amps']          = url.count('&')
        f['num_at']            = url.count('@')
        f['num_percent']       = url.count('%')
        f['num_digits']        = sum(c.isdigit() for c in url)
        f['num_letters']       = sum(c.isalpha() for c in url)
        f['digit_ratio']       = f['num_digits'] / max(len(url), 1)
        f['letter_ratio']      = f['num_letters'] / max(len(url), 1)
        special                = sum(c in '-_./?=&%#@:;+~$' for c in url)
        f['special_ratio']     = special / max(len(url), 1)
        f['alphanumeric_ratio']= (f['num_digits']+f['num_letters']) / max(len(url), 1)

        # 23-30: Structural flags
        f['has_ip']            = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', host) else 0
        f['has_https']         = 1 if p.scheme == 'https' else 0
        f['has_at']            = 1 if '@' in url else 0
        f['has_double_slash']  = 1 if '//' in url[8:] else 0
        f['has_dash_host']     = 1 if '-' in host else 0
        f['num_dashes_host']   = host.count('-')
        f['has_www']           = 1 if url_l.startswith('http://www.') or url_l.startswith('https://www.') else 0
        f['has_port']          = 1 if p.port and p.port not in (80, 443) else 0

        # 31-37: Domain structure
        parts                  = host.replace('www.', '').split('.')
        f['domain_length']     = len(parts[-2]) if len(parts) >= 2 else len(host)
        f['tld_length']        = len(parts[-1]) if parts else 0
        f['subdomain_count']   = max(len(parts) - 2, 0)
        f['has_subdomain']     = 1 if f['subdomain_count'] > 0 else 0
        f['many_subdomains']   = 1 if f['subdomain_count'] > 2 else 0
        f['domain_has_digits'] = 1 if any(c.isdigit() for c in (parts[-2] if len(parts) >= 2 else '')) else 0
        root_domain            = '.'.join(parts[-2:]) if len(parts) >= 2 else host
        f['is_known_legit']    = 1 if root_domain in LEGIT_DOMAIN_SET else 0

        # 38-40: TLD analysis
        suffix = '.' + parts[-1] if parts else ''
        f['has_susp_tld']      = 1 if suffix in SUSP_TLD_SET else 0
        f['tld_in_url_count']  = sum(url_l.count(t) for t in SUSP_TLD_SET)
        f['is_shortened']      = 1 if any(s in url_l for s in SHORTENERS) else 0

        # 41-44: Keyword analysis
        kw_hits                = sum(1 for kw in SUSP_KEYWORD_SET if kw in url_l)
        f['num_susp_kw']       = kw_hits
        f['has_susp_kw']       = 1 if kw_hits > 0 else 0
        path_kw                = sum(1 for kw in SUSP_KEYWORD_SET if kw in path.lower())
        f['num_susp_kw_path']  = path_kw
        f['has_susp_kw_path']  = 1 if path_kw > 0 else 0

        # 45-48: Brand impersonation
        brand_in_url           = sum(1 for b in BRAND_SET if b in url_l)
        f['brand_in_url']      = brand_in_url
        brand_in_host          = sum(1 for b in BRAND_SET if b in host)
        brand_legit            = any(b in root_domain for b in BRAND_SET)
        f['brand_host_imperson'] = 1 if brand_in_host > 0 and not brand_legit else 0
        brand_in_path          = sum(1 for b in BRAND_SET if b in path.lower())
        f['brand_in_path']     = 1 if brand_in_path > 0 else 0
        f['brand_not_host']    = 1 if (brand_in_url > 0 and not brand_legit) else 0

        # 49-52: Path analysis
        path_segs              = [s for s in path.split('/') if s]
        f['path_depth']        = len(path_segs)
        f['avg_seg_len']       = sum(len(s) for s in path_segs) / max(len(path_segs), 1)
        f['has_long_path']     = 1 if len(path) > 50 else 0
        f['has_php_ext']       = 1 if path.lower().endswith('.php') else 0

        # 53-54: Query parameters
        qparams                = parse_qs(qstr)
        f['num_qparams']       = len(qparams)
        f['has_susp_param']    = 1 if any(k in SUSP_PARAMS for k in qparams) else 0

        # 55-57: Entropy (randomness)
        if url:
            freq = Counter(url)
            entr = -sum((c/len(url)) * math.log2(c/len(url)) for c in freq.values())
            f['entropy']       = min(entr, 8.0)
            f['norm_entropy']  = entr / 8.0
        else:
            f['entropy'] = f['norm_entropy'] = 0.0

        # Max consecutive repeated character
        max_rep = run = 1
        for i in range(1, len(url)):
            run = run + 1 if url[i] == url[i-1] else 1
            max_rep = max(max_rep, run)
        f['max_repeat']        = min(max_rep / 10.0, 1.0)

    except Exception:
        # If URL can't be parsed, return zeros
        f = {k: 0 for k in [
            'url_length','host_length','path_length','query_length','host_ratio',
            'path_ratio','is_very_long','num_dots','num_hyphens','num_underscores',
            'num_slashes','num_qs','num_equals','num_amps','num_at','num_percent',
            'num_digits','num_letters','digit_ratio','letter_ratio','special_ratio',
            'alphanumeric_ratio','has_ip','has_https','has_at','has_double_slash',
            'has_dash_host','num_dashes_host','has_www','has_port','domain_length',
            'tld_length','subdomain_count','has_subdomain','many_subdomains',
            'domain_has_digits','is_known_legit','has_susp_tld','tld_in_url_count',
            'is_shortened','num_susp_kw','has_susp_kw','num_susp_kw_path',
            'has_susp_kw_path','brand_in_url','brand_host_imperson','brand_in_path',
            'brand_not_host','path_depth','avg_seg_len','has_long_path','has_php_ext',
            'num_qparams','has_susp_param','entropy','norm_entropy','max_repeat',
        ]}
    return f

FEATURE_NAMES = list(extract_url_features("https://google.com").keys())


def extract_feature_matrix(urls, verbose=True):
    rows = []
    n = len(urls)
    for i, u in enumerate(urls):
        rows.append(extract_url_features(u))
        if verbose and (i+1) % 5000 == 0:
            print(f"     Feature extraction: {i+1:,}/{n:,} ({(i+1)/n*100:.0f}%)")
    import pandas as pd
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df = df.fillna(0)
    return df


# ─── URL Model Trainer ────────────────────────────────────────────────────────
def train_url_model(urls=None, labels=None):
    print("\n" + "="*70)
    print("  URL MODEL TRAINING")
    print("="*70)

    if urls is None or labels is None:
        urls, labels = build_url_dataset()

    print(f"\n[2/6] Extracting URL features ({len(urls):,} URLs, {len(FEATURE_NAMES)} features)...")
    t0 = time.time()
    X_df = extract_feature_matrix(urls, verbose=True)
    print(f"     Done in {time.time()-t0:.1f}s")

    y = np.array(labels)
    X = X_df.values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n[3/6] Training URL ensemble model...")
    print(f"     Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Feature selection (top 45 features)
    n_select = min(45, X_train_scaled.shape[1])
    selector = SelectKBest(f_classif, k=n_select)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel  = selector.transform(X_test_scaled)
    print(f"     Selected {n_select} best features")

    # PCA embedding for fusion (64-dim)
    n_pca = min(EMBEDDING_DIM, X_train_sel.shape[1])
    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    pca.fit(X_train_sel)
    explained = pca.explained_variance_ratio_.sum()
    print(f"     PCA: {n_select} → {n_pca} dims  (explained: {explained:.1%})")

    # Ensemble
    rf  = RandomForestClassifier(
        n_estimators=300, max_depth=18, min_samples_split=5,
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
    )
    gb  = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=6,
        subsample=0.85, random_state=RANDOM_STATE
    )
    lr  = LogisticRegression(
        C=2.0, max_iter=1000, class_weight='balanced',
        solver='lbfgs', random_state=RANDOM_STATE
    )

    t1 = time.time()
    print("     Training RandomForest...")
    rf.fit(X_train_sel, y_train)
    print(f"     Training GradientBoosting... ({time.time()-t1:.0f}s so far)")
    gb.fit(X_train_sel, y_train)
    print(f"     Training LogisticRegression... ({time.time()-t1:.0f}s so far)")
    lr.fit(X_train_sel, y_train)

    model = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft'
    )
    model.fit(X_train_sel, y_train)
    print(f"     Total training time: {time.time()-t1:.1f}s")

    # Evaluate
    y_pred  = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1]

    metrics = {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'auc':       float(roc_auc_score(y_test, y_proba)),
        'train_size': len(X_train),
        'test_size':  len(X_test),
        'features':   len(FEATURE_NAMES),
        'pca_dims':   n_pca,
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  URL Model Results:")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-Score : {metrics['f1']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc']:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]}")
    print(f"      FN={cm[1,0]}  TP={cm[1,1]:,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

    # Save artifacts
    print(f"\n[4/6] Saving URL model to {URL_MODEL_DIR}...")
    joblib.dump(model,    os.path.join(URL_MODEL_DIR, 'model.joblib'),    compress=3)
    joblib.dump(scaler,   os.path.join(URL_MODEL_DIR, 'scaler.joblib'),   compress=3)
    joblib.dump(selector, os.path.join(URL_MODEL_DIR, 'selector.joblib'), compress=3)
    joblib.dump(pca,      os.path.join(URL_MODEL_DIR, 'pca.joblib'),      compress=3)

    with open(os.path.join(URL_MODEL_DIR, 'feature_names.json'), 'w') as fp:
        json.dump(FEATURE_NAMES, fp)
    with open(os.path.join(URL_MODEL_DIR, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)

    print("  URL model saved successfully!")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2:  TEXT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Text Training Data Templates ────────────────────────────────────────────
BANKS     = ["SBI","HDFC","ICICI","Axis","Kotak","Punjab National Bank","Bank of Baroda",
             "Canara Bank","Union Bank","Federal Bank","IndusInd Bank","Yes Bank"]
SERVICES  = ["PayPal","Google","Amazon","Netflix","Flipkart","Paytm","PhonePe",
             "Instagram","WhatsApp","LinkedIn","Twitter","Facebook","Airtel","Jio"]
AMOUNTS   = ["₹5,00,000","₹10,000","$10,000","$1,000,000","₹50,000","Rs. 25,000"]
HOURS     = ["24 hours","48 hours","6 hours","72 hours","12 hours"]
DATES     = ["31st December 2025","15th January 2026","28th February 2026","30th April 2026"]
PRODS     = ["iPhone 16 Pro","Samsung Galaxy S25","MacBook Pro","Dell Laptop","Smart TV"]
NAMES     = ["Raj","Priya","Amit","Sneha","Vikram","Ananya","Rahul","Pooja","Karthik","Meera"]


def _pick(*lst): return random.choice(lst[0] if len(lst) == 1 else lst)


PHISHING_TEMPLATES = [
    # Bank account alert
    lambda: (
        f"URGENT ALERT: Dear {_pick(BANKS)} customer, your account ending in "
        f"{random.randint(1000,9999)} has been temporarily SUSPENDED due to suspicious "
        f"login activity. Verify your identity immediately within {_pick(HOURS)} to avoid "
        f"permanent closure. Click the link below to verify: "
        f"http://secure-{_pick(BANKS).lower().replace(' ','-')}-verify.tk/login"
    ),
    # OTP / KYC scam
    lambda: (
        f"Your {_pick(SERVICES)} OTP for KYC verification is {random.randint(100000,999999)}. "
        f"DO NOT share this OTP with anyone. Your account will be blocked if KYC is not "
        f"completed within {_pick(HOURS)}. Complete KYC now: "
        f"http://kyc-update-{_pick(SERVICES).lower()}.xyz/verify?otp={random.randint(100000,999999)}"
    ),
    # Lottery / prize scam
    lambda: (
        f"Congratulations! You have been selected as the lucky winner of {_pick(AMOUNTS)} "
        f"in our {random.choice(['Monthly','Annual','Special'])} Draw. "
        f"To claim your prize, click here immediately: "
        f"http://prize-claim-{random.randint(1000,9999)}.online/winner "
        f"Offer expires in {_pick(HOURS)}. Claim NOW!"
    ),
    # Account suspension
    lambda: (
        f"Your {_pick(SERVICES)} account has been limited. We noticed unusual activity "
        f"from an unrecognized device. Your account will be permanently suspended on "
        f"{_pick(DATES)} unless you verify your credentials. "
        f"Update your information: http://{_pick(SERVICES).lower()}-account-verify.cf/secure"
    ),
    # Recharge / cashback scam
    lambda: (
        f"EXCLUSIVE OFFER! You have won a FREE recharge of {_pick(AMOUNTS)}. "
        f"This offer is valid only for the next {_pick(HOURS)}. "
        f"To claim, enter your mobile number and bank details at: "
        f"http://free-recharge-offer.tk/claim?ref={random.randint(10000,99999)}"
    ),
    # Job / HR phishing
    lambda: (
        f"Dear Applicant, Your application for the position of Software Engineer has been "
        f"shortlisted. Please complete your verification and submit your Aadhar, PAN, and "
        f"bank account details urgently at: "
        f"http://hr-hiring-portal-{random.randint(100,999)}.online/apply "
        f"Failure to submit within {_pick(HOURS)} will result in rejection."
    ),
    # Tax refund scam
    lambda: (
        f"Income Tax Department of India: Dear taxpayer, you are eligible for a tax refund "
        f"of {_pick(AMOUNTS)}. Kindly verify your bank account details to receive the refund "
        f"within 24 hours. Visit: http://incometax-refund-claim.xyz/verify "
        f"Note: Do not ignore this message. Deadline: {_pick(DATES)}"
    ),
    # Credit card / EMI scam
    lambda: (
        f"ALERT: Your {_pick(BANKS)} Credit Card ending {random.randint(1000,9999)} "
        f"has exceeded its limit. An EMI payment of {_pick(AMOUNTS)} is overdue. "
        f"To avoid legal action and CIBIL impact, pay immediately at: "
        f"http://{_pick(BANKS).lower().replace(' ','-')}-payment.ml/dues "
        f"Failure to act within {_pick(HOURS)} may result in card suspension."
    ),
    # Parcel / delivery scam
    lambda: (
        f"Your parcel from {_pick(SERVICES)} could not be delivered. "
        f"Tracking: PK{random.randint(10000000,99999999)}IN. "
        f"To reschedule delivery, confirm your address and pay a ₹35 redelivery fee at: "
        f"http://delivery-rescheduling-{random.randint(100,999)}.site/confirm"
    ),
    # Fake security warning
    lambda: (
        f"SECURITY WARNING: We detected that your {_pick(SERVICES)} password was compromised "
        f"in a data breach. Hackers may have access to your account. "
        f"Reset your password IMMEDIATELY to secure your account: "
        f"http://{_pick(SERVICES).lower()}-password-reset.xyz/urgent?token={_rand_alnum()}"
    ),
]

LEGIT_TEMPLATES = [
    # Meeting request
    lambda: (
        f"Hi {_pick(NAMES)}, I hope you're doing well. I wanted to schedule a meeting to "
        f"discuss the project progress. Could we connect on {_pick(DATES)} at "
        f"{random.randint(10,18)}:00? Please confirm your availability. "
        f"Best regards, {_pick(NAMES)}"
    ),
    # Shipping update
    lambda: (
        f"Your order #{random.randint(10000000,99999999)} has been shipped! "
        f"Estimated delivery: {_pick(DATES)}. Track your package at "
        f"https://www.{random.choice(['amazon.in','flipkart.com','myntra.com'])}/track "
        f"Thank you for shopping with us."
    ),
    # Newsletter
    lambda: (
        f"Good morning! Here is your daily digest for {_pick(DATES)}. "
        f"Today's top stories: New government policy announced on digital infrastructure, "
        f"Tech sector sees record investments, and more. "
        f"Read more at https://www.{random.choice(['thehindu.com','ndtv.com','indianexpress.com'])}"
    ),
    # Professional update
    lambda: (
        f"Hi {_pick(NAMES)}, just a quick update on the quarterly report. "
        f"The numbers are looking great with a {random.randint(10,40)}% increase YoY. "
        f"I'll share the full presentation before {_pick(DATES)}. "
        f"Let me know if you need anything. Cheers, {_pick(NAMES)}"
    ),
    # Payment receipt / legal
    lambda: (
        f"Invoice #{random.randint(1000,9999)} - Payment Received. "
        f"Dear {_pick(NAMES)}, we have received your payment of {_pick(AMOUNTS)}. "
        f"Your subscription/service has been renewed until {_pick(DATES)}. "
        f"For any queries, contact support@{random.choice(['zoho.com','freshworks.com','razorpay.com'])}"
    ),
    # Internal HR announcement
    lambda: (
        f"Team Announcement: We are pleased to welcome {_pick(NAMES)} to our team as "
        f"Senior {random.choice(['Developer','Designer','Analyst','Manager'])}. "
        f"Please join me in extending a warm welcome. "
        f"The team lunch is scheduled for {_pick(DATES)}."
    ),
    # Product release / update
    lambda: (
        f"We're excited to announce the latest update to {_pick(PRODS)}! "
        f"New features include improved performance, enhanced security, and a refreshed interface. "
        f"Download the update from the official store or visit "
        f"https://www.{random.choice(['apple.com','microsoft.com','google.com'])}/updates"
    ),
    # Educational / course enrollment
    lambda: (
        f"Hi {_pick(NAMES)}, your enrollment for the course is confirmed. "
        f"Start date: {_pick(DATES)}. Access your course materials at "
        f"https://www.{random.choice(['nptel.ac.in','coursera.org','udemy.com'])}. "
        f"Reach out if you have any questions. Good luck!"
    ),
    # Travel booking
    lambda: (
        f"Your booking is confirmed! PNR: {random.randint(1000000000,9999999999)}. "
        f"Journey: {random.choice(['Mumbai','Delhi','Bangalore','Chennai'])} to "
        f"{random.choice(['Hyderabad','Pune','Kolkata','Jaipur'])} on {_pick(DATES)}. "
        f"Check your booking at https://www.irctc.co.in/nget/train-search"
    ),
    # Technical support
    lambda: (
        f"Hi {_pick(NAMES)}, this is a follow-up on your support ticket #{random.randint(10000,99999)}. "
        f"Our team has resolved the issue you reported. The fix has been deployed to production. "
        f"Please verify at your convenience and let us know if you face any further issues."
    ),
]


def build_text_dataset(phishing_count=TEXT_PHISHING_COUNT, legit_count=TEXT_LEGIT_COUNT):
    print(f"\n[1/4] Generating text training data ({phishing_count} phishing + {legit_count} legitimate)...")
    t0 = time.time()

    phishing_texts, legit_texts = [], []

    for _ in range(phishing_count):
        phishing_texts.append(random.choice(PHISHING_TEMPLATES)())

    for _ in range(legit_count):
        legit_texts.append(random.choice(LEGIT_TEMPLATES)())

    texts  = phishing_texts + legit_texts
    labels = [1] * phishing_count + [0] * legit_count

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    print(f"     Done in {time.time()-t0:.1f}s  |  Total: {len(texts):,} samples")
    return list(texts), list(labels)


# ─── Text Model Trainer ───────────────────────────────────────────────────────
def train_text_model(texts=None, labels=None):
    print("\n" + "="*70)
    print("  TEXT MODEL TRAINING")
    print("="*70)

    if texts is None or labels is None:
        texts, labels = build_text_dataset()

    y = np.array(labels)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n[2/4] Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,        # log-scaling of term frequency
        strip_accents='unicode',
        analyzer='word',
        stop_words='english',
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)
    print(f"     Vocabulary size: {len(vectorizer.vocabulary_):,}")

    print(f"\n[3/4] Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        C=5.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=RANDOM_STATE,
    )
    t1 = time.time()
    classifier.fit(X_train, y_train)
    print(f"     Training time: {time.time()-t1:.2f}s")

    # Evaluate
    y_pred  = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'auc':       float(roc_auc_score(y_test, y_proba)),
        'train_size': len(X_train_raw),
        'test_size':  len(X_test_raw),
        'vocab_size': len(vectorizer.vocabulary_),
    }

    print(f"\n  Text Model Results:")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-Score : {metrics['f1']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

    # Save
    print(f"\n[4/4] Saving text model to {TEXT_MODEL_DIR}...")
    joblib.dump(vectorizer,  os.path.join(TEXT_MODEL_DIR, 'vectorizer.joblib'),  compress=3)
    joblib.dump(classifier,  os.path.join(TEXT_MODEL_DIR, 'classifier.joblib'), compress=3)
    with open(os.path.join(TEXT_MODEL_DIR, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)

    print("  Text model saved successfully!")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='PhishGuard Model Trainer')
    parser.add_argument('--url',  action='store_true', help='Train URL model only')
    parser.add_argument('--text', action='store_true', help='Train text model only')
    args = parser.parse_args()

    train_all  = not args.url and not args.text
    train_url  = args.url  or train_all
    train_text = args.text or train_all

    total_start = time.time()
    results = {}

    if train_url:
        urls, url_labels = build_url_dataset()
        results['url']  = train_url_model(urls, url_labels)

    if train_text:
        texts, text_labels = build_text_dataset()
        results['text'] = train_text_model(texts, text_labels)

    elapsed = time.time() - total_start
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    if 'url' in results:
        print(f"  URL Model  - Accuracy: {results['url']['accuracy']:.2%}  AUC: {results['url']['auc']:.4f}")
    if 'text' in results:
        print(f"  Text Model - Accuracy: {results['text']['accuracy']:.2%}  AUC: {results['text']['auc']:.4f}")
    print(f"  Total time : {elapsed:.1f}s")
    print(f"  Models saved to: {MODELS_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
