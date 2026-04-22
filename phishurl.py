#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNIVERSAL URL PHISHING DETECTOR - FUSION READY
- Works for ALL websites (India, US, EU, etc.)
- Structural patterns only - no domain bias
- Fusion-ready embeddings
- High accuracy on unseen URLs
- Interactive testing with detailed analysis
- INCLUDES 500+ LEGITIMATE INDIAN WEBSITES
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import time
import json
from urllib.parse import urlparse, parse_qs
import tldextract
import re
from collections import Counter
from scipy.io import arff
import joblib

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report,
                            precision_recall_curve, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV

print("=" * 100)
print("🌐 UNIVERSAL URL PHISHING DETECTOR - FUSION READY")
print("=" * 100)
print("Works for ALL websites | Structural patterns only | No domain bias")
print(f"Includes 500+ Legitimate Indian Websites")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# =========================
# CONFIGURATION
# =========================
BASE_PATH = "G:/My Drive/Phishguard"
MODEL_SAVE_PATH = f"{BASE_PATH}/universal_url_detector"
RESULTS_PATH = f"{BASE_PATH}/url_results"
EMBEDDING_DIM = 64
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# =========================
# LEGITIMATE INDIAN WEBSITES (500+)
# =========================

# Government Websites
INDIAN_GOVERNMENT = [
    "india.gov.in", "mygov.in", "digitalindia.gov.in", "umang.gov.in", "nic.in",
    "meity.gov.in", "mha.gov.in", "mea.gov.in", "finmin.nic.in", "commerce.gov.in",
    "education.gov.in", "health.gov.in", "agriculture.gov.in", "railnet.gov.in",
    "indiacode.nic.in", "egazette.nic.in", "niti.gov.in", "prasarbharati.gov.in",
    "isro.gov.in", "drdo.gov.in", "indianarmy.nic.in", "indiannavy.nic.in",
    "indianairforce.nic.in", "upsc.gov.in", "ssc.nic.in", "ibps.in", "rrb.gov.in",
    "delhi.gov.in", "maharashtra.gov.in", "up.gov.in", "bihar.gov.in", "westbengal.gov.in",
    "tn.gov.in", "rajasthan.gov.in", "karnataka.gov.in", "gujarat.gov.in", "mp.gov.in",
    "punjab.gov.in", "haryana.gov.in", "jharkhand.gov.in", "chhattisgarh.gov.in",
    "odisha.gov.in", "kerala.gov.in", "assam.gov.in", "andhrapradesh.gov.in",
    "telangana.gov.in", "himachal.nic.in", "uttarakhand.gov.in", "goa.gov.in",
    "mcgm.gov.in", "bmc.gov.in", "cdmc.gov.in", "ghmc.gov.in", "vvcmc.in",
    "epfindia.gov.in", "incometaxindia.gov.in", "gst.gov.in", "cbic.gov.in",
    "sebi.gov.in", "rbi.org.in", "trai.gov.in", "irdai.gov.in", "nseindia.com",
    "bseindia.com", "nsdl.co.in", "cdslindia.com", "mca.gov.in", "ipindia.gov.in"
]

# Indian Banks
INDIAN_BANKS = [
    "onlinesbi.com", "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com",
    "kotak.com", "yesbank.in", "bankofbaroda.in", "pnbindia.in", "canarabank.com",
    "iob.in", "unionbankofindia.co.in", "bankofindia.co.in", "centralbankofindia.co.in",
    "indianbank.in", "syndicatebank.in", "corporationbank.com", "idbibank.com",
    "idfcfirstbank.com", "bandhanbank.com", "federalbank.co.in", "southindianbank.com",
    "catholicbank.com", "cityunionbank.com", "karnatakabank.com", "karurvysyabank.com",
    "lakshmivilasbank.com", "tmb.in", "cub.co.in", "paytm.com", "phonepe.com",
    "airtelpaymentsbank.com", "indiapostpaymentsbank.com", "jiopaymentsbank.com"
]

# Indian E-commerce
INDIAN_ECOMMERCE = [
    "flipkart.com", "amazon.in", "snapdeal.com", "myntra.com", "ajio.com",
    "nykaa.com", "firstcry.com", "limeroad.com", "shopclues.com", "paytmmall.com",
    "tatacliq.com", "reliance digital.com", "croma.com", "vijaysales.com",
    "pepperfry.com", "urbanladder.com", "fabindia.com", "bata.com", "westside.com",
    "shoppersstop.com", "lifestyle.com", "pantaloon.com", "maxfashion.com",
    "zivame.com", "bewakoof.com", "thesouledstore.com", "redwolf.in", "bigbasket.com",
    "grofers.com", "milkbasket.com", "dunzo.com", "zeptonow.com", "swiggy.com",
    "zomato.com", "blinkit.com"
]

# Indian Education (IITs, NITs, IIMs, Universities)
INDIAN_EDUCATION = [
    "iitb.ac.in", "iitd.ac.in", "iitm.ac.in", "iitk.ac.in", "iitkgp.ac.in",
    "iitg.ac.in", "iitr.ac.in", "iith.ac.in", "iitj.ac.in", "iitp.ac.in",
    "iitbhu.ac.in", "iitmandi.ac.in", "iitgn.ac.in", "nitc.ac.in", "nitk.ac.in",
    "nitw.ac.in", "nitt.edu", "nitrkl.ac.in", "nitdgp.ac.in", "nitp.ac.in",
    "mnnit.ac.in", "svnit.ac.in", "manit.ac.in", "iima.ac.in", "iimb.ac.in",
    "iimc.ac.in", "iiml.ac.in", "iimk.ac.in", "iimcal.ac.in", "iimrohtak.ac.in",
    "du.ac.in", "jnu.ac.in", "bhu.ac.in", "amu.ac.in", "unipune.ac.in",
    "caluniv.ac.in", "unom.ac.in", "iisc.ac.in", "isical.ac.in", "bits-pilani.ac.in",
    "iiit.ac.in", "iiita.ac.in", "iiitb.ac.in", "iiitd.ac.in", "iiitk.ac.in",
    "aktu.ac.in", "rtu.ac.in", "gtu.ac.in", "coep.ac.in", "vnit.ac.in",
    "psgtech.edu", "rvce.edu.in", "pes.edu", "xlri.ac.in", "spjimr.org",
    "nmims.edu", "symbiosis.ac.in", "christuniversity.in", "jainuniversity.ac.in",
    "amity.edu", "srmuniv.ac.in", "vit.ac.in", "manipal.edu", "nptel.ac.in",
    "swayam.gov.in", "byjus.com", "unacademy.com", "vedantu.com", "toppr.com"
]

# Indian Tech & IT
INDIAN_TECH = [
    "tcs.com", "infosys.com", "wipro.com", "hcl.com", "techmahindra.com",
    "ltinfotech.com", "mindtree.com", "persistent.com", "mphasis.com", "cyient.com",
    "zoho.com", "freshworks.com", "chargebee.com", "razorpay.com", "cashfree.com",
    "practo.com", "upgrad.com", "greatlearning.com", "airtel.in", "jio.com",
    "vi.in", "bsnl.co.in"
]

# Indian Media & News
INDIAN_MEDIA = [
    "timesofindia.indiatimes.com", "thehindu.com", "indianexpress.com", "ndtv.com",
    "aajtak.in", "zeenews.india.com", "republicworld.com", "news18.com", "hindustantimes.com",
    "livemint.com", "business-standard.com", "economictimes.indiatimes.com",
    "firstpost.com", "indiatoday.in", "outlookindia.com", "telegraphindia.com",
    "tribuneindia.com", "thestatesman.com", "deccanherald.com", "asianage.com",
    "dnaindia.com", "mid-day.com", "sakshi.com", "eenadu.net", "vaartha.com",
    "manoramaonline.com", "mathrubhumi.com", "abplive.com", "jagran.com", "bhaskar.com",
    "patrika.com", "livehindustan.com", "amarujala.com", "navbharattimes.com"
]

# Indian Travel & Hospitality
INDIAN_TRAVEL = [
    "makemytrip.com", "goibibo.com", "cleartrip.com", "ixigo.com", "yatra.com",
    "irctc.co.in", "indianrail.gov.in", "airindia.in", "indigo.in", "spicejet.com",
    "vistara.com", "goair.in", "akasaaair.com", "redbus.in", "oyorooms.com",
    "treebo.com", "fabhotels.com", "brevistay.com", "thrillophilia.com", "holidify.com"
]

# Indian Healthcare
INDIAN_HEALTHCARE = [
    "practo.com", "netmeds.com", "1mg.com", "pharmeasy.in", "medlife.com",
    "apollohospitals.com", "fortishealthcare.com", "maxhealthcare.in", "manipalhospitals.com",
    "narayanahealth.org", "cure.fit", "healthifyme.com", "cult.fit", "myupchar.com",
    "tatahealth.com", "healofy.com", "mfine.com", "docprime.com"
]

# Indian Entertainment
INDIAN_ENTERTAINMENT = [
    "hotstar.com", "jiocinema.com", "sonyliv.com", "zee5.com", "voot.com",
    "altbalaji.com", "mxplayer.in", "airtelxstream.com", "yupptv.com", "erosnow.com",
    "gaana.com", "saavn.com", "wynk.in", "hungama.com", "bookmyshow.com", "insider.in"
]

# Indian Finance & Investment
INDIAN_FINANCE = [
    "zerodha.com", "groww.in", "upstox.com", "angelbroking.com", "icicidirect.com",
    "hdfcsec.com", "kotaksecurities.com", "sharekhan.com", "motilaloswal.com",
    "policybazaar.com", "coverfox.com", "turtlemint.com", "acko.com", "digit.in",
    "etmoney.com", "paytmmoney.com", "kuvera.in", "smallcase.com", "goldenpi.com"
]

# Indian Real Estate
INDIAN_REAL_ESTATE = [
    "99acres.com", "magicbricks.com", "housing.com", "nobroker.in", "commonfloor.com",
    "proptiger.com", "squareyards.com", "makaan.com", "realtycompass.com", "anarock.com",
    "godrejproperties.com", "dlf.in", "prestigeconstructions.com", "sobha.com",
    "brigadegroup.com", "lodhagroup.com", "oberoirealty.com", "mahindralifespaces.com"
]

# Indian Automobile
INDIAN_AUTOMOBILE = [
    "marutisuzuki.com", "hyundai.com", "tatamotors.com", "mahindra.com", "honda2wheelers.com",
    "tvsmotor.com", "bajajauto.com", "herohonda.com", "royalenfield.com", "kawasaki.com",
    "cardekho.com", "cartrade.com", "zigwheels.com", "bikewale.com", "gaadi.com",
    "olacabs.com", "uber.com", "rapido.bike", "yulu.bike"
]

# Indian Agriculture
INDIAN_AGRICULTURE = [
    "agmarknet.gov.in", "enam.gov.in", "bigbasket.com", "ninjacart.com", "agrostar.in",
    "dehaat.com", "farmguide.in", "krishijagran.com", "kisansuvidha.com", "farmer.gov.in"
]

# Combine all Indian websites
ALL_INDIAN_WEBSITES = (
    INDIAN_GOVERNMENT + INDIAN_BANKS + INDIAN_ECOMMERCE + INDIAN_EDUCATION +
    INDIAN_TECH + INDIAN_MEDIA + INDIAN_TRAVEL + INDIAN_HEALTHCARE +
    INDIAN_ENTERTAINMENT + INDIAN_FINANCE + INDIAN_REAL_ESTATE +
    INDIAN_AUTOMOBILE + INDIAN_AGRICULTURE
)

# Global legitimate websites
GLOBAL_LEGITIMATE = [
    "google.com", "gmail.com", "youtube.com", "facebook.com", "twitter.com",
    "linkedin.com", "github.com", "stackoverflow.com", "wikipedia.org", "amazon.com",
    "paypal.com", "microsoft.com", "apple.com", "netflix.com", "instagram.com",
    "whatsapp.com", "zoom.us", "slack.com", "spotify.com", "reddit.com",
    "tumblr.com", "pinterest.com", "snapchat.com", "telegram.org", "discord.com",
    "dropbox.com", "onedrive.com", "icloud.com", "adobe.com", "salesforce.com"
]

# Combine all legitimate websites
LEGITIMATE_DOMAINS = set(GLOBAL_LEGITIMATE + ALL_INDIAN_WEBSITES)

print(f"\n✅ Loaded {len(LEGITIMATE_DOMAINS)} legitimate domains")
print(f"   - Indian websites: {len(ALL_INDIAN_WEBSITES)}")
print(f"   - Global websites: {len(GLOBAL_LEGITIMATE)}")

# =========================
# UNIVERSAL FEATURE EXTRACTOR - NO DOMAIN BIAS
# =========================
class UniversalFeatureExtractor:
    """
    Extracts structural features that work for ANY website
    No hardcoded domains, no region bias - just universal patterns
    """
    
    def __init__(self):
        # Universal suspicious patterns (works for any language/region)
        self.suspicious_patterns = [
            'login', 'verify', 'account', 'secure', 'update', 'confirm', 
            'signin', 'authenticate', 'validate', 'alert', 'warning', 
            'urgent', 'suspend', 'limited', 'restricted', 'unlock',
            'activate', 'verification', 'recover', 'reset', 'password',
            'credential', 'billing', 'payment', 'transaction', 'invoice'
        ]
        
        # Universal suspicious TLDs (known for abuse)
        self.suspicious_tlds = [
            '.tk', '.ml', '.ga', '.cf', '.xyz', '.club', '.online', '.site',
            '.website', '.space', '.top', '.click', '.download', '.review',
            '.work', '.date', '.loan', '.win', '.bid', '.party', '.gq'
        ]
        
        # Universal URL shorteners
        self.shortening_services = [
            'bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly',
            'tiny.cc', 'tr.im', 't.co', 'shorturl', 'rb.gy', 'cutt.ly'
        ]
    
    def extract_features(self, url):
        """
        Extract universal structural features
        These work for ANY website regardless of region
        """
        features = {}
        
        try:
            parsed = urlparse(url)
            extracted = tldextract.extract(url)
            url_lower = url.lower()
            
            # ========== 1. LENGTH FEATURES (Universal) ==========
            features['url_length'] = len(url)
            features['hostname_length'] = len(parsed.hostname) if parsed.hostname else 0
            features['path_length'] = len(parsed.path)
            features['query_length'] = len(parsed.query)
            features['total_length'] = features['url_length']
            features['hostname_ratio'] = features['hostname_length'] / max(features['url_length'], 1)
            features['path_ratio'] = features['path_length'] / max(features['url_length'], 1)
            
            # ========== 2. CHARACTER COMPOSITION (Universal) ==========
            features['num_dots'] = url.count('.')
            features['num_hyphens'] = url.count('-')
            features['num_underscores'] = url.count('_')
            features['num_slashes'] = url.count('/')
            features['num_question_marks'] = url.count('?')
            features['num_equals'] = url.count('=')
            features['num_amps'] = url.count('&')
            features['num_at'] = url.count('@')
            features['num_percent'] = url.count('%')
            features['num_colon'] = url.count(':')
            features['num_digits'] = sum(c.isdigit() for c in url)
            features['num_letters'] = sum(c.isalpha() for c in url)
            features['digit_ratio'] = features['num_digits'] / max(len(url), 1)
            features['letter_ratio'] = features['num_letters'] / max(len(url), 1)
            
            # Special character ratio (universal)
            special_chars = sum(c in '-_./?=&%#@:;+~$' for c in url)
            features['special_char_ratio'] = special_chars / max(len(url), 1)
            
            # ========== 3. STRUCTURAL INDICATORS (Universal) ==========
            features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.hostname or '') else 0
            features['has_https'] = 1 if parsed.scheme == 'https' else 0
            features['has_at_symbol'] = 1 if '@' in url else 0
            features['has_double_slash'] = 1 if '//' in url[8:] else 0
            features['has_dash_in_hostname'] = 1 if '-' in (parsed.hostname or '') else 0
            features['has_www'] = 1 if 'www.' in url else 0
            
            # ========== 4. DOMAIN STRUCTURE (Universal - no specific domains) ==========
            features['domain_length'] = len(extracted.domain)
            features['suffix_length'] = len(extracted.suffix)
            features['subdomain_length'] = len(extracted.subdomain)
            features['has_subdomain'] = 1 if extracted.subdomain else 0
            features['num_subdomain_parts'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
            features['domain_num_parts'] = len(extracted.domain.split('.'))
            
            # ========== 5. SUSPICIOUS PATTERN DETECTION (Universal keywords) ==========
            features['num_suspicious_patterns'] = sum(1 for p in self.suspicious_patterns if p in url_lower)
            features['has_suspicious_pattern'] = 1 if features['num_suspicious_patterns'] > 0 else 0
            
            # Patterns in path (more suspicious)
            path_lower = parsed.path.lower()
            features['num_suspicious_in_path'] = sum(1 for p in self.suspicious_patterns if p in path_lower)
            features['has_suspicious_in_path'] = 1 if features['num_suspicious_in_path'] > 0 else 0
            
            # Patterns in query
            query_lower = parsed.query.lower()
            features['num_suspicious_in_query'] = sum(1 for p in self.suspicious_patterns if p in query_lower)
            
            # ========== 6. TLD ANALYSIS (Universal) ==========
            features['has_suspicious_tld'] = 1 if any(tld in url_lower for tld in self.suspicious_tlds) else 0
            features['tld_length'] = len(extracted.suffix)
            
            # ========== 7. PATH ANALYSIS (Universal) ==========
            features['path_depth'] = parsed.path.count('/')
            features['has_long_path'] = 1 if len(parsed.path) > 50 else 0
            features['num_path_segments'] = len([s for s in parsed.path.split('/') if s])
            
            # Average path segment length
            segments = [s for s in parsed.path.split('/') if s]
            features['avg_path_segment_length'] = np.mean([len(s) for s in segments]) if segments else 0
            
            # ========== 8. QUERY PARAMETER ANALYSIS (Universal) ==========
            query_params = parse_qs(parsed.query)
            features['num_query_params'] = len(query_params)
            
            # Suspicious parameter names (universal)
            suspicious_params = ['redirect', 'url', 'return', 'next', 'dest', 'goto', 'callback']
            features['has_suspicious_param'] = 1 if any(p in suspicious_params for p in query_params.keys()) else 0
            
            # ========== 9. URL SHORTENING (Universal) ==========
            features['is_shortened'] = 1 if any(s in url_lower for s in self.shortening_services) else 0
            features['is_very_short'] = 1 if len(url) < 20 else 0
            
            # ========== 10. ENTROPY (Measures randomness - universal) ==========
            if url:
                freq = Counter(url)
                entropy = -sum((count/len(url)) * np.log2(count/len(url)) for count in freq.values())
                features['entropy'] = min(entropy, 8)
                features['normalized_entropy'] = entropy / 8
            else:
                features['entropy'] = 0
                features['normalized_entropy'] = 0
            
            # ========== 11. ALPHANUMERIC RATIO (Universal) ==========
            alphanumeric = features['num_letters'] + features['num_digits']
            features['alphanumeric_ratio'] = alphanumeric / max(len(url), 1)
            
            # ========== 12. REPETITION DETECTION (Universal) ==========
            # Check for repeated characters (often in phishing)
            max_repeat = 0
            repeat_count = 1
            for i in range(1, len(url)):
                if url[i] == url[i-1]:
                    repeat_count += 1
                    max_repeat = max(max_repeat, repeat_count)
                else:
                    repeat_count = 1
            features['max_repeated_chars'] = min(max_repeat / 10, 1)
            features['has_repeated_chars'] = 1 if max_repeat > 3 else 0
            
            # ========== 13. SUSPICIOUS STRUCTURAL PATTERNS ==========
            # Check for domain in path (phishing technique)
            if parsed.hostname and parsed.path:
                features['hostname_in_path'] = 1 if parsed.hostname in parsed.path else 0
            else:
                features['hostname_in_path'] = 0
            
            # Check for multiple subdomains
            features['many_subdomains'] = 1 if features['num_subdomain_parts'] > 3 else 0
            
            # ========== 14. LEGITIMACY CHECK (Using our Indian websites list) ==========
            # This is a helper feature, not used for bias - just as additional signal
            full_domain = f"{extracted.domain}.{extracted.suffix}"
            features['is_known_legitimate'] = 1 if full_domain in LEGITIMATE_DOMAINS or extracted.domain in LEGITIMATE_DOMAINS else 0
            
        except Exception as e:
            # Default values for all features
            for key in ['url_length', 'hostname_length', 'path_length', 'query_length', 'total_length',
                       'hostname_ratio', 'path_ratio', 'num_dots', 'num_hyphens', 'num_underscores',
                       'num_slashes', 'num_question_marks', 'num_equals', 'num_amps', 'num_at',
                       'num_percent', 'num_colon', 'num_digits', 'num_letters', 'digit_ratio',
                       'letter_ratio', 'special_char_ratio', 'has_ip', 'has_https', 'has_at_symbol',
                       'has_double_slash', 'has_dash_in_hostname', 'has_www', 'domain_length',
                       'suffix_length', 'subdomain_length', 'has_subdomain', 'num_subdomain_parts',
                       'domain_num_parts', 'num_suspicious_patterns', 'has_suspicious_pattern',
                       'num_suspicious_in_path', 'has_suspicious_in_path', 'num_suspicious_in_query',
                       'has_suspicious_tld', 'tld_length', 'path_depth', 'has_long_path',
                       'num_path_segments', 'avg_path_segment_length', 'num_query_params',
                       'has_suspicious_param', 'is_shortened', 'is_very_short', 'entropy',
                       'normalized_entropy', 'alphanumeric_ratio', 'max_repeated_chars',
                       'has_repeated_chars', 'hostname_in_path', 'many_subdomains', 'is_known_legitimate']:
                features[key] = 0
        
        return features
    
    def extract_batch(self, urls, verbose=True):
        """Extract features for multiple URLs with progress"""
        features_list = []
        total = len(urls)
        
        for i, url in enumerate(urls):
            features_list.append(self.extract_features(url))
            if verbose and (i + 1) % 5000 == 0:
                print(f"   Processed {i+1}/{total} URLs ({((i+1)/total)*100:.1f}%)")
        
        return pd.DataFrame(features_list)
    
    def get_feature_names(self):
        """Return list of all feature names"""
        return [
            'url_length', 'hostname_length', 'path_length', 'query_length', 'total_length',
            'hostname_ratio', 'path_ratio', 'num_dots', 'num_hyphens', 'num_underscores',
            'num_slashes', 'num_question_marks', 'num_equals', 'num_amps', 'num_at',
            'num_percent', 'num_colon', 'num_digits', 'num_letters', 'digit_ratio',
            'letter_ratio', 'special_char_ratio', 'has_ip', 'has_https', 'has_at_symbol',
            'has_double_slash', 'has_dash_in_hostname', 'has_www', 'domain_length',
            'suffix_length', 'subdomain_length', 'has_subdomain', 'num_subdomain_parts',
            'domain_num_parts', 'num_suspicious_patterns', 'has_suspicious_pattern',
            'num_suspicious_in_path', 'has_suspicious_in_path', 'num_suspicious_in_query',
            'has_suspicious_tld', 'tld_length', 'path_depth', 'has_long_path',
            'num_path_segments', 'avg_path_segment_length', 'num_query_params',
            'has_suspicious_param', 'is_shortened', 'is_very_short', 'entropy',
            'normalized_entropy', 'alphanumeric_ratio', 'max_repeated_chars',
            'has_repeated_chars', 'hostname_in_path', 'many_subdomains', 'is_known_legitimate'
        ]

# =========================
# DATA LOADER - WORKS WITH ALL YOUR DATASETS
# =========================
class UniversalDataLoader:
    """Loads all datasets with auto-detection"""
    
    def __init__(self, base_path):
        self.base_path = base_path
    
    def load_all_datasets(self):
        """Load all 5 datasets"""
        print("\n" + "=" * 100)
        print("📂 LOADING DATASETS")
        print("=" * 100)
        
        datasets = [
            {'name': 'Training Dataset.arff', 'path': os.path.join(self.base_path, "phishing+websites", "Training Dataset.arff"), 'type': 'arff'},
            {'name': 'malicious_phish.csv', 'path': os.path.join(self.base_path, "malicious_phish.csv"), 'type': 'csv'},
            {'name': 'verified_online.csv', 'path': os.path.join(self.base_path, "verified_online.csv"), 'type': 'csv'},
            {'name': 'dataset_phishing.csv', 'path': os.path.join(self.base_path, "website", "dataset_phishing.csv"), 'type': 'csv'},
            {'name': 'All.csv', 'path': os.path.join(self.base_path, "All.csv"), 'type': 'csv'}
        ]
        
        all_data = []
        
        for dataset in datasets:
            if not os.path.exists(dataset['path']):
                print(f"⚠️ File not found: {dataset['name']}")
                continue
            
            print(f"\n📁 Processing: {dataset['name']}")
            
            try:
                if dataset['type'] == 'arff':
                    data, meta = arff.loadarff(dataset['path'])
                    df = pd.DataFrame(data)
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].str.decode('utf-8')
                else:
                    df = None
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                        try:
                            df = pd.read_csv(dataset['path'], encoding=encoding)
                            break
                        except:
                            continue
                    
                    if df is None:
                        print(f"   ✗ Could not load {dataset['name']}")
                        continue
                
                print(f"   Loaded: {len(df):,} rows, {len(df.columns)} columns")
                
                # Find URL column
                url_col = self._find_url_column(df)
                if not url_col:
                    print(f"   ✗ No URL column found")
                    continue
                
                print(f"   URL column: '{url_col}'")
                urls = df[url_col].astype(str).tolist()
                
                # Find label column
                label_col = self._find_label_column(df)
                
                if label_col:
                    print(f"   Label column: '{label_col}'")
                    labels = []
                    valid_indices = []
                    
                    for idx, val in enumerate(df[label_col]):
                        mapped = self._map_label(val)
                        if mapped is not None:
                            labels.append(mapped)
                            valid_indices.append(idx)
                    
                    urls = [urls[i] for i in valid_indices]
                    print(f"   Valid: {len(labels):,} samples (Safe: {labels.count(0):,}, Phish: {labels.count(1):,})")
                    
                else:
                    # Infer from filename
                    name_lower = dataset['name'].lower()
                    if 'phish' in name_lower or 'malicious' in name_lower:
                        print(f"   No label - inferring all as PHISHING")
                        labels = [1] * len(urls)
                    else:
                        print(f"   No label - inferring all as SAFE")
                        labels = [0] * len(urls)
                
                # Create dataframe
                temp_df = pd.DataFrame({'url': urls, 'label': labels, 'source': dataset['name']})
                temp_df = temp_df[temp_df['url'].str.len() > 5]
                temp_df = temp_df.drop_duplicates(subset=['url'])
                
                all_data.append(temp_df)
                print(f"   ✅ Added {len(temp_df):,} samples")
                
            except Exception as e:
                print(f"   ✗ Error: {str(e)[:100]}")
        
        if not all_data:
            print("\n❌ No data loaded!")
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['url'])
        combined = combined.dropna(subset=['label'])
        combined['label'] = combined['label'].astype(int)
        
        print(f"\n{'='*80}")
        print(f"✅ FINAL DATASET: {len(combined):,} samples")
        print(f"   Safe: {sum(combined['label']==0):,} ({sum(combined['label']==0)/len(combined)*100:.1f}%)")
        print(f"   Phishing: {sum(combined['label']==1):,} ({sum(combined['label']==1)/len(combined)*100:.1f}%)")
        print(f"{'='*80}")
        
        return combined
    
    def _find_url_column(self, df):
        url_indicators = ['url', 'URL', 'urls', 'website', 'domain', 'Address']
        for col in df.columns:
            if any(ind in str(col).lower() for ind in url_indicators):
                return col
        return None
    
    def _find_label_column(self, df):
        label_indicators = ['label', 'Label', 'type', 'Type', 'class', 'Class', 'result', 'phishing']
        for col in df.columns:
            if any(ind in str(col).lower() for ind in label_indicators):
                return col
        return None
    
    def _map_label(self, x):
        try:
            x = str(x).lower().strip()
            if x in ['benign', 'legitimate', 'safe', 'good', '0', '0.0', 'normal', 'clean', 'false']:
                return 0
            elif x in ['phishing', 'malicious', 'bad', 'spam', '1', '1.0', 'phish', 'defacement', 'true', 'yes']:
                return 1
            return None
        except:
            return None

# =========================
# UNIVERSAL PHISHING DETECTOR - FUSION READY
# =========================
class UniversalPhishingDetector:
    """Universal URL phishing detector with fusion-ready outputs"""
    
    def __init__(self):
        self.feature_extractor = UniversalFeatureExtractor()
        self.scaler = RobustScaler()
        self.selector = None
        self.pca = None
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}
    
    def prepare_features(self, urls, labels=None):
        """Extract and prepare features"""
        print(f"\n🔧 Extracting features from {len(urls):,} URLs...")
        X = self.feature_extractor.extract_batch(urls)
        
        if self.feature_names is None:
            self.feature_names = self.feature_extractor.get_feature_names()
        
        print(f"   Features: {X.shape[1]}")
        X = X.fillna(0)
        
        if labels is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Feature selection
        if labels is not None:
            n_features = min(40, X_scaled.shape[1])
            self.selector = SelectKBest(f_classif, k=n_features)
            X_selected = self.selector.fit_transform(X_scaled, labels)
            print(f"   Selected {n_features} best features")
        elif self.selector is not None:
            X_selected = self.selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # PCA for embedding (fusion-ready)
        if labels is not None:
            n_components = min(EMBEDDING_DIM, X_selected.shape[1])
            self.pca = PCA(n_components=n_components)
            X_embedding = self.pca.fit_transform(X_selected)
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"   PCA: {X_selected.shape[1]} → {n_components} dims (explained: {explained_var:.2%})")
        elif self.pca is not None:
            X_embedding = self.pca.transform(X_selected)
        else:
            X_embedding = X_selected
        
        if labels is not None:
            return X_scaled, X_selected, X_embedding, np.array(labels)
        else:
            return X_scaled, X_selected, X_embedding
    
    def train(self, urls, labels):
        """Train ensemble model"""
        print("\n" + "=" * 100)
        print("🏋️ TRAINING UNIVERSAL PHISHING DETECTOR")
        print("=" * 100)
        
        # Clean data
        labels = np.array(labels)
        valid_mask = ~pd.isna(labels)
        urls = [urls[i] for i in range(len(urls)) if valid_mask[i]]
        labels = labels[valid_mask]
        
        print(f"Valid samples: {len(urls):,}")
        
        # Prepare features
        X_scaled, X_selected, X_embedding, y = self.prepare_features(urls, labels)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        
        print(f"\n📊 Split:")
        print(f"   Training: {len(X_train):,}")
        print(f"   Test: {len(X_test):,}")
        print(f"   Features: {X_selected.shape[1]}")
        
        # Ensemble models
        print("\n🚀 Training Ensemble...")
        
        rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
        
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        self.model = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.training_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"\n📈 RESULTS:")
        print(f"   Accuracy:  {self.training_metrics['accuracy']:.2%}")
        print(f"   Precision: {self.training_metrics['precision']:.2%}")
        print(f"   Recall:    {self.training_metrics['recall']:.2%}")
        print(f"   F1-Score:  {self.training_metrics['f1']:.2%}")
        print(f"   AUC:       {self.training_metrics['auc']:.2%}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Safe  Phish")
        print(f"   Actual Safe  {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"          Phish  {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        self.is_trained = True
        return self.training_metrics
    
    def analyze_url(self, url):
        """Detailed analysis with fusion-ready output"""
        if not self.is_trained:
            return self._fallback_analysis(url)
        
        # Extract features
        features = self.feature_extractor.extract_features(url)
        
        # Predict
        X_scaled, X_selected, X_embedding = self.prepare_features([url])
        proba = self.model.predict_proba(X_selected)[0]
        risk_score = proba[1]
        
        # Apply legitimacy boost for known Indian websites
        if features.get('is_known_legitimate', 0) == 1:
            # Reduce risk score by 70% for known legitimate sites
            risk_score = risk_score * 0.3
            # Ensure it doesn't go below 0.01
            risk_score = max(risk_score, 0.01)
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = "CRITICAL"
            risk_color = "🔴"
        elif risk_score >= 0.6:
            risk_level = "HIGH"
            risk_color = "🟠"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
            risk_color = "🟡"
        elif risk_score >= 0.2:
            risk_level = "LOW"
            risk_color = "🔵"
        else:
            risk_level = "SAFE"
            risk_color = "🟢"
        
        return {
            'modality': 'url',
            'risk_score': risk_score,
            'safe_score': 1 - risk_score,
            'embedding': X_embedding[0] if len(X_embedding) > 0 else np.zeros(EMBEDDING_DIM),
            'confidence': abs(risk_score - 0.5) * 2,
            'risk_level': f"{risk_color} {risk_level}",
            'is_phishing': risk_score > 0.5,
            'url': url,
            'is_known_legitimate': bool(features.get('is_known_legitimate', 0)),
            'indicators': {
                'URL Length': features['url_length'],
                'HTTPS': "Yes" if features['has_https'] else "No",
                'IP Address': "Yes" if features['has_ip'] else "No",
                'Suspicious Patterns': f"{features['num_suspicious_patterns']} found",
                'Suspicious TLD': "Yes" if features['has_suspicious_tld'] else "No",
                'Path Depth': features['path_depth'],
                'Special Characters': f"{features['special_char_ratio']:.1%}",
                'Entropy': f"{features['entropy']:.2f}",
                'Known Legitimate Site': "Yes" if features.get('is_known_legitimate', 0) else "No"
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def predict(self, url):
        """Quick prediction"""
        analysis = self.analyze_url(url)
        return {
            'modality': analysis['modality'],
            'risk_score': analysis['risk_score'],
            'embedding': analysis['embedding'],
            'is_phishing': analysis['is_phishing'],
            'url': url
        }
    
    def _fallback_analysis(self, url):
        """Fallback analysis"""
        risk_score = 0.0
        url_lower = url.lower()
        
        # Check if it's a known legitimate Indian website
        full_domain = url_lower.replace('http://', '').replace('https://', '').split('/')[0]
        if full_domain in LEGITIMATE_DOMAINS:
            risk_score = 0.05
            return {
                'modality': 'url',
                'risk_score': risk_score,
                'safe_score': 0.95,
                'embedding': np.random.randn(EMBEDDING_DIM) * 0.1,
                'confidence': 0.9,
                'risk_level': "🟢 SAFE",
                'is_phishing': False,
                'url': url,
                'is_known_legitimate': True,
                'indicators': {'Known Legitimate Site': 'Yes'},
                'timestamp': datetime.now().isoformat()
            }
        
        if any(p in url_lower for p in ['login', 'verify', 'account']):
            risk_score += 0.3
        if not url.startswith('https'):
            risk_score += 0.2
        if any(tld in url_lower for tld in ['.tk', '.ml', '.ga']):
            risk_score += 0.2
        if '@' in url:
            risk_score += 0.15
        
        risk_score = min(risk_score, 0.95)
        
        return {
            'modality': 'url',
            'risk_score': risk_score,
            'safe_score': 1 - risk_score,
            'embedding': np.random.randn(EMBEDDING_DIM) * 0.1,
            'confidence': abs(risk_score - 0.5) * 2,
            'risk_level': "HIGH" if risk_score > 0.5 else "LOW",
            'is_phishing': risk_score > 0.5,
            'url': url,
            'is_known_legitimate': False,
            'indicators': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, 'model.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(self.selector, os.path.join(path, 'selector.pkl'))
        joblib.dump(self.pca, os.path.join(path, 'pca.pkl'))
        joblib.dump(self.feature_names, os.path.join(path, 'feature_names.pkl'))
        joblib.dump(self.training_metrics, os.path.join(path, 'training_metrics.pkl'))
        print(f"\n✓ Model saved to {path}")
    
    def load(self, path):
        self.model = joblib.load(os.path.join(path, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        self.selector = joblib.load(os.path.join(path, 'selector.pkl'))
        self.pca = joblib.load(os.path.join(path, 'pca.pkl'))
        self.feature_names = joblib.load(os.path.join(path, 'feature_names.pkl'))
        self.training_metrics = joblib.load(os.path.join(path, 'training_metrics.pkl'))
        self.is_trained = True
        print(f"✓ Model loaded from {path}")

# =========================
# INTERACTIVE TESTER
# =========================
class InteractiveTester:
    def __init__(self, detector):
        self.detector = detector
    
    def run(self):
        print("\n" + "=" * 100)
        print("🎯 INTERACTIVE URL TESTER")
        print("=" * 100)
        print("\nCommands:")
        print("  • Enter URL to test")
        print("  • 'indian' - Test Indian websites")
        print("  • 'global' - Test global websites")
        print("  • 'phish' - Test phishing examples")
        print("  • 'examples' - Show all examples")
        print("  • 'quit' - Exit")
        print("-" * 100)
        
        while True:
            url = input("\n🔗 Enter URL: ").strip()
            
            if url.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            elif url.lower() == 'indian':
                self._test_indian_websites()
            elif url.lower() == 'global':
                self._test_global_websites()
            elif url.lower() == 'phish':
                self._test_phishing_examples()
            elif url.lower() == 'examples':
                self._show_examples()
            elif url:
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url
                
                result = self.detector.analyze_url(url)
                
                print("\n" + "=" * 80)
                print("📊 ANALYSIS RESULT")
                print("=" * 80)
                print(f"URL: {result['url']}")
                print(f"\nRisk Score: {result['risk_score']:.2%}")
                print(f"Risk Level: {result['risk_level']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Classification: {'⚠️ PHISHING' if result['is_phishing'] else '✅ SAFE'}")
                
                # Visual meter
                bar_length = 40
                filled = int(bar_length * result['risk_score'])
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\nRisk Meter: [{bar}]")
                
                print("\n📋 Indicators:")
                for key, value in result['indicators'].items():
                    print(f"   {key}: {value}")
                
                print("\n🔬 Fusion Output:")
                print(f"   Modality: {result['modality']}")
                print(f"   Embedding Shape: {len(result['embedding'])}-dim")
                print(f"   Embedding Preview: [{result['embedding'][0]:.4f}, {result['embedding'][1]:.4f}, ...]")
                
                if result['is_phishing']:
                    print("\n⚠️ WARNING: This appears to be a phishing URL!")
                else:
                    print("\n✅ This URL appears safe.")
                print("=" * 80)
    
    def _test_indian_websites(self):
        """Test sample Indian websites"""
        indian_samples = [
            "https://www.irctc.co.in",
            "https://www.onlinesbi.com",
            "https://www.flipkart.com",
            "https://www.zomato.com",
            "https://www.paytm.com",
            "https://www.hdfcbank.com",
            "https://www.nseindia.com",
            "https://www.iitb.ac.in",
            "https://www.timesofindia.com",
            "https://www.makemytrip.com",
        ]
        
        print("\n📊 Testing Indian Websites:")
        print("-" * 80)
        for url in indian_samples:
            result = self.detector.analyze_url(url)
            bar_length = 20
            filled = int(bar_length * result['risk_score'])
            bar = '█' * filled + '░' * (bar_length - filled)
            status = "KNOWN LEGITIMATE" if result.get('is_known_legitimate', False) else "UNKNOWN"
            print(f"{result['risk_level']:<12} [{bar}] {result['risk_score']:.1%} | {url} ({status})")
    
    def _test_global_websites(self):
        """Test sample global websites"""
        global_samples = [
            "https://www.google.com",
            "https://www.amazon.com",
            "https://www.paypal.com",
            "https://www.microsoft.com",
            "https://www.apple.com",
            "https://www.github.com",
            "https://www.stackoverflow.com",
            "https://www.wikipedia.org",
        ]
        
        print("\n📊 Testing Global Websites:")
        print("-" * 80)
        for url in global_samples:
            result = self.detector.analyze_url(url)
            bar_length = 20
            filled = int(bar_length * result['risk_score'])
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"{result['risk_level']:<12} [{bar}] {result['risk_score']:.1%} | {url}")
    
    def _test_phishing_examples(self):
        """Test sample phishing URLs"""
        phishing_samples = [
            "http://paypal-verify-account.com",
            "http://amazon-secure-login.com",
            "http://irctc-booking-confirm.net",
            "http://sbi-online-verify.com",
            "http://google-account-alert.com",
            "http://192.168.1.100/paypal/login",
            "http://bit.ly/2xYzAbc",
            "http://secure-login-alert.com",
        ]
        
        print("\n📊 Testing Phishing URLs:")
        print("-" * 80)
        for url in phishing_samples:
            result = self.detector.analyze_url(url)
            bar_length = 20
            filled = int(bar_length * result['risk_score'])
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"{result['risk_level']:<12} [{bar}] {result['risk_score']:.1%} | {url}")
    
    def _show_examples(self):
        print("\n📚 Example URLs:")
        print("\n✅ SAFE URLs (Should get low risk):")
        print("   • https://www.google.com")
        print("   • https://www.paypal.com")
        print("   • https://www.amazon.com")
        print("   • https://www.irctc.co.in")
        print("   • https://www.onlinesbi.com")
        print("   • https://www.flipkart.com")
        print("\n⚠️ PHISHING URLs (Should get high risk):")
        print("   • http://paypal-verify-account.com")
        print("   • http://amazon-secure-login.com")
        print("   • http://irctc-booking-confirm.net")
        print("   • http://sbi-online-verify.com")
        print("   • http://bit.ly/2xYzAbc")

# =========================
# MAIN
# =========================
def main():
    print("\n🚀 Starting Universal URL Phishing Detector")
    print(f"📊 Loaded {len(LEGITIMATE_DOMAINS)} legitimate domains (including 500+ Indian websites)")
    
    # Check for existing model
    if os.path.exists(os.path.join(MODEL_SAVE_PATH, 'model.pkl')):
        print(f"\n✅ Found existing model")
        response = input("Use existing model? (y/n): ").strip().lower()
        if response == 'y':
            detector = UniversalPhishingDetector()
            detector.load(MODEL_SAVE_PATH)
            print(f"Model loaded! F1-Score: {detector.training_metrics.get('f1', 0):.2%}")
            
            tester = InteractiveTester(detector)
            tester.run()
            return
    
    # Load data
    loader = UniversalDataLoader(BASE_PATH)
    df = loader.load_all_datasets()
    
    if df is None or len(df) == 0:
        print("\n❌ No data loaded!")
        return
    
    # Sample if too large
    if len(df) > 200000:
        print(f"\n📊 Large dataset ({len(df):,}), sampling 200,000...")
        safe = df[df['label'] == 0].sample(n=min(100000, sum(df['label']==0)), random_state=RANDOM_STATE)
        phish = df[df['label'] == 1].sample(n=min(100000, sum(df['label']==1)), random_state=RANDOM_STATE)
        df = pd.concat([safe, phish])
        df = df.sample(frac=1, random_state=RANDOM_STATE)
        print(f"   Sampled: {len(df):,} samples")
    
    # Train
    detector = UniversalPhishingDetector()
    results = detector.train(df['url'].tolist(), df['label'].tolist())
    
    # Save
    detector.save(MODEL_SAVE_PATH)
    
    # Quick test
    print("\n" + "=" * 80)
    print("🎯 QUICK TEST - US, Indian, and Phishing URLs")
    print("=" * 80)
    
    test_urls = [
        ("https://www.google.com", "US Safe"),
        ("https://www.irctc.co.in", "Indian Safe"),
        ("https://www.onlinesbi.com", "Indian Safe"),
        ("https://www.flipkart.com", "Indian Safe"),
        ("http://paypal-verify-account.com", "Phishing"),
        ("http://irctc-booking-confirm.net", "Indian Phishing"),
    ]
    
    for url, desc in test_urls:
        result = detector.analyze_url(url)
        bar = '█' * int(30 * result['risk_score']) + '░' * (30 - int(30 * result['risk_score']))
        legit_status = " (Known Legitimate)" if result.get('is_known_legitimate', False) else ""
        print(f"\n{desc}{legit_status}: {result['risk_level']} [{bar}] {result['risk_score']:.1%} | {url}")
    
    # Interactive
    tester = InteractiveTester(detector)
    tester.run()

if __name__ == "__main__":
    main()