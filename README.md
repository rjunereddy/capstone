# 🛡️ PhishGuard AI: Multi-Modal Phishing Detection System

![Version](https://img.shields.io/badge/version-3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Architecture](https://img.shields.io/badge/architecture-Cloud%20Multi--Modal-success)
![Status](https://img.shields.io/badge/status-Production%20Ready-success)

PhishGuard AI is an advanced, next-generation machine learning pipeline designed to detect complex zero-day phishing attacks. Traditional security tools rely on URL blacklists or basic text scanning, which fail against modern orchestrated attacks. PhishGuard introduces a **5-Modality Fusion Engine** that mathematically correlates signals across URL, Text, Visual Layout, Audio (Vishing), and Video (Deepfake) to definitively neutralize threats.

---

## ✨ Key Features

- **Multi-Modal AI Fusion**: Simultaneously evaluates 5 critical data pathways instead of just 1.
- **Cross-Modal Correlation**: A proprietary mathematical fusion engine that boosts risk scores if multiple modalities trigger (e.g., Suspicious Video combined with Suspicious Text).
- **SHAP-Based Explainable AI (XAI)**: Demystifies the "Black Box" of machine learning by visually highlighting *exactly why* an attack was flagged (e.g., `High-Risk Domain TLD: +0.28`).
- **Interactive Simulation Lab**: A built-in Chrome Extension dashboard that allows cybersecurity analysts to fire safe, simulated multi-modal attacks directly against the live cloud engine to track threat correlation.

---

## 🧠 The 5 Modalities

1. **🌐 URL Structure (Machine Learning)**
   - Extracts 57 lexical and host-based features.
   - Detects brand impersonation (e.g., `paypal-auth.tk` vs `paypal.com`), excessive entropy, and hostile Top Level Domains (.tk, .ml).
2. **📝 Text & NLP (Machine Learning)**
   - TF-IDF vectorization mapping 1,600+ semantic tokens.
   - Detects artificial urgency ("account suspended") and credential harvesting scripts.
3. **🖼️ Visual Layout (Heuristic/ResNet Pattern)**
   - Detects "Brand Cloning"—where the visual arrangement of a website perfectly mimics a trusted entity but on an unverified network.
4. **🎙️ Audio / Vishing (Phonetic Analysis)**
   - Protects against Voice Phishing by scanning embedded audio traps for social engineering terminology.
5. **📹 Video / Deepfake (Structural Integrity)**
   - Scans embedded media for synthetic manipulation metadata indicating face-swapping or deepfake presence.

---

## 🏗️ System Architecture

PhishGuard operates on a robust, decoupled Client-Server architecture:

### 1. Browser Extension (The Client)
Built on **Chrome Manifest V3**, the extension runs silently in the background. It dynamically intercepts page loads, extracts the DOM (Document Object Model) vectors, and transmits them to the cloud. The UI provides a premium dashboard representing the Live Gauge Score and Modality Threat Bars.

### 2. Cloud Machine Learning API (The Server)
The heavy AI inference is decoupled from the user's browser to preserve local compute. Hosted via **Flask/Gunicorn** on a cloud infrastructure (Render AWS/GCP), the server processes incoming arrays, normalizes the embeddings, executes the mathematical fusion, and returns a comprehensive JSON threat report.

---

## 🚀 Installation & Deployment

### 1. End-User Installation (Zero-Configuration)
Because the heavy AI inference engine is hosted securely in the cloud, users do NOT need Python or any local processing power. They simply install the Chrome extension:
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** in the top right.
3. Click **Load unpacked** and select the `/extension` directory inside this repository.
4. The extension automatically connects to the live Render Cloud API (`https://phishguard-ai-r5wa.onrender.com`).

### 2. Cloud Server Deployment (For Developers)
The backend is natively configured for automated Render Cloud deployment.
1. Connect this GitHub repository to [Render.com](https://render.com).
2. **Build Command:** `pip install -r requirements.txt`.
3. **Start Command:** `gunicorn server:app --timeout 60 --workers 2`.
4. **Environment Variable:** Set `PYTHONUTF8 = 1` for log compatibility.
5. Once deployed, the server runs 24/7. Update the `SERVER_URL` inside `extension/background.js` if your Render URL ever changes.

---

## 💻 Usage & Demonstration

### The Simulation Sandbox
1. Open the PhishGuard Chrome Extension.
2. Navigate to the **🧪 Simulator** tab.
3. Select a scenario (e.g., **Critical Multi-Modal Attack**).
4. The extension will artificially construct a malicious payload, hit the live Cloud Server, and graphically spin up the 5 modality meters, displaying the Cross-Modal Conflict multiplier and SHAP impacts.

---

## 🛠️ Technology Stack
- **Backend & ML Pipeline**: Python 3.10+, Scikit-Learn, NumPy, SciPy, Joblib, Flask, Gunicorn.
- **Frontend & UI**: JavaScript (ES6), HTML5, Vanilla CSS, Chrome Manifest V3 APIs.
- **Algorithms**: Random Forest, Multi-Layer Perceptron (MLP), TF-IDF Vectorization, Principal Component Analysis (PCA), Cosine Similarity Correlation.

---
*Developed for Academic Capstone Demonstration.*
