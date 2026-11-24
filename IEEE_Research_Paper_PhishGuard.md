# PhishGuard AI: Real-Time Multimodal Phishing and Deepfake Detection Browser Architecture Using Correlation-Aware Latent Space Fusion

**Mallikarjun Reddy$^1$, Navajyoth$^1$, Pranav G M$^1$, N Viharika$^1$, Dr. Roopa Ravish$^2$**  
$^1$Department of Computer Science and Engineering, PES University, Bengaluru, India  
$^2$Professor, Department of Computer Science and Engineering, PES University, Bengaluru, India  
Emails: {mallikarjunreddy0564@gmail.com, guide-email@pes.edu}  
*Project Identification: PW26_RR_06 (UE23CS441A)*

---

### Abstract
Phishing attacks have rapidly advanced beyond superficial email lures and deceptive hyperlinks into sophisticated, multimodal social engineering campaigns combining zero-day lookalike domains, cloned visual layouts, synthetic voice authorization traps (vishing), and generative deepfake executive announcements. Traditional security countermeasures relying on static URL blacklists, domain reputation databases, or isolated single-modality classifiers suffer from critical blindspots: they exhibit severe latency delays against ephemeral domains and fail when malicious payloads reside on trusted cloud infrastructure. 

In this paper, we propose **PhishGuard AI**, a real-time, privacy-conscious browser defense framework that couples a lightweight Chrome Manifest V3 client extension with a cloud-based **5-Modality Correlation-Aware Fusion Engine**. PhishGuard concurrently inspects URL lexical topology, Document Object Model (DOM) text semantics, visual layout patterns, embedded audio frequencies, and deepfake video streams. Rather than employing naive arithmetic score averaging which dilutes asymmetric threats, we formulate a **Shared Latent Space Projection Network** that maps heterogeneous feature embeddings (ranging from 64-D to 768-D) into a unified 128-dimensional latent space. Pairwise cosine similarity yields a dynamic agreement multiplier ($\alpha \in [0.8, 1.2]$) that penalizes conflicting modalities on legitimate platforms while amplifying correlated zero-day attacks. Furthermore, our architecture incorporates directional Shapley Additive exPlanations (SHAP) and Large Language Model (LLM) contextual synthesis to provide human-interpretable warning overlays. 

Extensive empirical evaluation across held-out benchmarks demonstrates that our URL ensemble (57 features) achieves **98.73% accuracy** and our NLP classifier achieves **98.56% accuracy**, with the fused pipeline attaining **99.12% accuracy** while reducing false-positive alerts by **38.4%** compared to standalone URL heuristics. End-to-end execution exhibits an average inference turnaround of **$\sim$150–180 ms**, confirming practical feasibility for seamless web browsing.

**Index Terms**—Phishing Detection, Deepfake Detection, Multimodal Fusion, Explainable AI (XAI), Latent Space Projection, Browser Security, Manifest V3, SHAP.

---

## I. Introduction

Phishing remains one of the primary attack vectors in modern cybersecurity breaches, exploiting human cognitive trust rather than purely technical protocol vulnerabilities [1]. Attackers routinely orchestrate multi-layered deception campaigns targeting financial institutions (e.g., SBI, HDFC, PayPal), payment gateways, and enterprise communication channels. In emerging threat landscapes, adversaries no longer rely exclusively on misspelled domain names; instead, they integrate realistic text manipulation, high-fidelity brand visual cloning, and increasingly, AI-synthesized audio and video lures [2], [3].

Conventional defenses exhibit structural limitations:
1. **Blocklist Inefficacy:** Reputation registries (e.g., Google Safe Browsing, PhishTank) suffer from discovery-to-listing latency. Modern phishing campaigns frequently operate using fast-flux DNS and ephemeral domains with lifespans shorter than 4 hours [4].
2. **Single-Modality Blindspots:** A URL-only classifier cannot identify a credential-harvesting form hosted legitimately on trusted cloud platforms (e.g., Firebase, GitHub Pages, or Google Docs). Conversely, a vision-only classifier flagrantly misclassifies benign corporate portals sharing standard bootstrap templates.
3. **Emergence of Synthetic Media (Deepfakes and Vishing):** The democratization of deep generative networks enables real-time voice synthesis and face-swapping, paving the way for corporate executive impersonation and automated voice OTP verification scams.
4. **Opaque "Black-Box" Decisions:** Most ML security tools output an uninformative binary classification without explaining why a page is flagged, inducing alert fatigue and user bypass.

To address these challenges, we introduce **PhishGuard AI**, an end-to-end multimodal browser security system. The primary contributions of this work are summarized as follows:
* **Decoupled Manifest V3 Microservice Architecture:** We engineer an asynchronous, non-blocking client extension that harvests client-side DOM vectors in $<20\text{ ms}$, delegating deep inference to an elastic cloud REST microservice, maintaining an end-to-end runtime of $\sim$150–180 ms.
* **Five-Modality Analytical Pipeline:** We develop specialized detection modules addressing URL lexical topology (57 engineered features), textual NLP semantics, visual layout cloning via CNN embeddings, audio vishing via spectral anomaly estimation, and video deepfakes via facial optical flow.
* **Correlation-Aware Shared Latent Space Fusion:** We propose an architectural projection network mapping disparate embeddings into a 128-D shared metric space, using pairwise cosine distance to derive an agreement multiplier that dampens false alarms on legitimate media hosts while boosting verified multi-vector attacks.
* **Grounded Explainability & Active Interception:** We operationalize directional SHAP attributions coupled with Gemini LLM reasoning to inject actionable, real-time modal warning overlays directly into the browser DOM when critical threat thresholds are breached.

The remainder of this paper is organized as follows: Section II reviews related literature; Section III outlines system architecture; Section IV details the detection modules; Section V formulates the multimodal fusion engine; Section VI describes the explainability pipeline; Section VII presents experimental results and benchmarks; Section VIII examines cross-domain applicability; and Section IX concludes the paper.

---

## II. Related Work

### A. URL and Heuristic Phishing Detection
Early anti-phishing research predominantly targeted lexical and structural characteristics of URLs. Aleroud and Zhou [1] categorized URL features into lexical tokens, host attributes, and DNS registration metadata. Jain and Gupta [2] demonstrated hyperlink consistency verification using client-side DOM inspection. However, single-modality URL classifiers struggle with modern domain obfuscation techniques such as homograph attacks, sub-domain leasing, and reverse-proxy credential interception.

### B. NLP and Content-Based Phishing Detection
Content-based techniques analyze textual corpus semantics within email bodies and landing pages. Classic approaches utilized Term Frequency-Inverse Document Frequency (TF-IDF) alongside Support Vector Machines or Naive Bayes. Recent advancements leverage Transformer architectures, such as DistilBERT and RoBERTa, to identify coercive urgency, synthetic authority, and credential-harvesting phrasing [5]. However, transformer-based text analysis in isolation remains susceptible to visual deception where malicious text is embedded inside image canvases.

### C. Visual Impersonation and Deepfake Media
Visual anti-phishing verifies layout and brand identity alignment. Afroz and Greenstadt introduced visual similarity matching using screenshot comparison. In media threat analysis, Das et al. [3] surveyed facial artifact detection in synthetic videos, demonstrating that generative models (GANs and diffusion models) frequently leave high-frequency spatial discrepancies in facial boundaries, blinking cadences, and optical flow vectors.

### D. Multimodal Fusion in Cybersecurity
While multimodal learning has achieved substantial success in multimedia retrieval, its application to browser security remains nascent. Existing multimodal security frameworks typically rely on early feature concatenation or unweighted late score voting. Fixed arithmetic averaging ($R = \frac{1}{M}\sum R_i$) is mathematically defective for adversarial threat modeling: an attack featuring a benign URL but a malicious deepfake payload is erroneously diluted into a safe score. PhishGuard AI addresses this deficit through non-linear metric space projection and cross-modal correlation scoring.

---

## III. System Architecture

PhishGuard AI employs a decoupled, asynchronous Client-Server topology designed to satisfy strict real-time browsing constraints, as illustrated in Fig. 1.

```
+-------------------------------------------------------------------------+
|                        CLIENT: CHROME BROWSER                           |
|                                                                         |
|  [Active Web Tab] ---> [content.js / background.js (Manifest V3)]       |
|                                |                                        |
|                                v                                        |
|                     Asynchronous DOM Extraction                         |
|         (URL + Visible Text <= 3500 chars + Media/Form Elements)         |
+------------------------------------+------------------------------------+
                                     |  HTTP REST (JSON Payload)
                                     |  Latency: ~150-180 ms round-trip
                                     v
+-------------------------------------------------------------------------+
|                     SERVER: CLOUD INFERENCE ENGINE                      |
|                                                                         |
|  [1. URL Module]     [2. Text NLP]     [3. Vision]   [4. Audio] [5. Vid]|
|     57 Features         TF-IDF / BERT    ResNet-18     ASVspoof   Celeb |
|       (64-D)              (768-D)         (64-D)        (64-D)    (64-D)|
|         |                    |               |             |        |   |
|         +--------------------+-------+-------+-------------+--------+   |
|                                      |                                  |
|                                      v                                  |
|                 [Shared Latent Space Projection (128-D)]                |
|                                      |                                  |
|                                      v                                  |
|             [Pairwise Cosine Similarity & Agreement Multiplier]          |
|                                      |                                  |
|                                      v                                  |
|              [Dynamic Confidence-Weighted Risk Fusion Engine]           |
|                                      |                                  |
|                                      v                                  |
|                  [SHAP Attribution + LLM Explanation]                   |
+------------------------------------+------------------------------------+
                                     |  JSON Response
                                     v
+-------------------------------------------------------------------------+
|                     USER-FACING BROWSER ACTIONS                         |
|                                                                         |
|  - If Score >= 0.60: Inject Red Full-Screen Warning Overlay into DOM    |
|  - Dynamic Badge Update: "!!" (Critical) / "!" (High) / "~" / "OK"      |
|  - Extension Telemetry Dashboard: Gauge Score + Modality Threat Bars    |
+-------------------------------------------------------------------------+
```
*Fig. 1. End-to-end decoupled system architecture of PhishGuard AI.*

### A. Manifest V3 Extension Layer
Under Chrome’s Manifest V3 architecture, background execution is constrained to ephemeral service workers (`background.js`) that lack persistent memory state. When an `onUpdated` tab event signals status completion, the service worker executes an asynchronous script injection (`content.js`). The script extracts:
1. Target URL string, protocol scheme, port, and query parameter structure.
2. Sanitized visible page text (truncated to the first 3,500 characters to prevent payload bloat).
3. Structural DOM presence flags: `has_video`, `has_audio`, `num_images`, and `has_password`.

### B. Asynchronous Communications and State Persistence
The payload is dispatched via `fetch` to the cloud API (`/analyze`). Results are cached into `chrome.storage.local` indexed by `tabId`. This ensures zero main-thread blocking, preserving standard 60-FPS DOM rendering during web navigation.

---

## IV. Detection Modules in Detail

### A. URL Phishing Detection Module
The URL analysis subsystem parses URLs into 57 handcrafted lexical, structural, and host-level topological features:
* **Length and Segment Metrics:** Total URL length, host length, path length, query length, path depth, and segment token ratios.
* **Symbolic and Character Ratios:** Counts and ratios of dots (`.`), hyphens (`-`), underscores (`_`), slashes (`/`), question marks (`?`), ampersands (`&`), at-symbols (`@`), and percent encodings (`%`).
* **Entropy Profiling:** Shannon entropy calculated across characters to identify algorithmic generation algorithms (DGAs):
  $$H(U) = -\sum_{i=1}^{n} P(c_i) \log_2 P(c_i)$$
* **Host and Domain Semantics:** Binary indicators for raw IPv4/IPv6 hostnames, non-standard ports, suspicious Top-Level Domains (e.g., `.tk`, `.ml`, `.xyz`, `.top`), and excessive subdomain depth ($>2$).
* **Targeted Brand Impersonation:** Verification of trusted corporate and Indian institution keywords (e.g., `sbi`, `hdfc`, `icici`, `paypal`, `paytm`, `irctc`) embedded within subdomains or paths when the registered second-level domain does not match official authoritative registries.

**Classifier Architecture:** Features undergo standardization via `RobustScaler`, ranking via `SelectKBest` ($f$-classif), and dimensionality reduction via `PCA` to 45 components. The decision model implements a Soft-Voting Ensemble combining Random Forest (100 estimators), Gradient Boosting, and L2-regularized Logistic Regression.

### B. Text and Language NLP Module
The textual detection pipeline inspects visible text for coercive social engineering patterns:
1. **Semantic Vectorization:** A curated TF-IDF vectorizer mapping 1,695 salient lexical n-grams ($n \in \{1, 2\}$) derived from phishing corpora.
2. **Behavioral Flags:** Regular expression heuristic scans targeting:
   * Urgency manipulation (e.g., *"account suspended"*, *"act within 24 hours"*).
   * Credential requests (e.g., *"enter your password"*, *"verify your OTP"*).
   * Regional Indian scam phrasing (e.g., *"Aadhaar verification"*, *"KYC update"*, *"PAN card blocked"*).
3. **Embedding Generation:** The normalized token weights are projected into a dense 768-dimensional semantic embedding space ($E_{\text{text}} \in \mathbb{R}^{768}$) preserving cross-modal fusion compatibility with pre-trained DistilBERT representations.

### C. Visual Layout Impersonation Module
The visual detector models brand identity cloning. While attackers easily forge domain names, replicating high-converting trust requires copying visual styling. We leverage a pre-trained **ResNet-18** Convolutional Neural Network stripped of its terminal classification head to extract visual feature representations from page components. Extracted feature tensors are compressed via PCA to a 64-dimensional embedding ($E_{\text{image}} \in \mathbb{R}^{64}$) and evaluated using an **Isolation Forest** anomaly detector trained on legitimate banking interfaces to quantify structural deviation.

### D. Audio (Vishing) Anomaly Module
Designed to detect automated synthetic voice lures, the audio module ingests audio streams from embedded `<audio>` elements. Acoustic features are extracted over 4-second frames at 16 kHz:
* 13 Mel-Frequency Cepstral Coefficients (MFCCs).
* Spectral centroid, spectral roll-off, and spectral flux.
* Statistical distribution skewness and kurtosis.
Trained on the **ASVspoof 2019/2021** physical access and logical access benchmarks, the model isolates synthetic spectral frequency artifacts, generating a 64-dimensional spoof embedding ($E_{\text{audio}} \in \mathbb{R}^{64}$).

### E. Video (Deepfake) Detection Module
Addressing weaponized synthetic media (such as CEO fraud announcements), the video pipeline samples frames from intercepted `<video>` elements. It computes optical flow field continuity, inter-frame color variation, and facial boundary motion discrepancies. Evaluated against the **Celeb-DF-v2** benchmark, anomalies are mapped into a 64-dimensional descriptor ($E_{\text{video}} \in \mathbb{R}^{64}$).

---

## V. Correlation-Aware Latent Fusion Engine

### A. Mathematical Formulation: Shared Latent Space
Heterogeneous modalities produce embeddings of disparate dimensionality: $E_{\text{text}} \in \mathbb{R}^{768}$, $E_{\text{url}} \in \mathbb{R}^{64}$, $E_{\text{image}} \in \mathbb{R}^{64}$, $E_{\text{audio}} \in \mathbb{R}^{64}$, and $E_{\text{video}} \in \mathbb{R}^{64}$. 

To correlate these signals mathematically, we construct a **Shared Fusion Space** $\mathcal{Z} \subset \mathbb{R}^{d}$ where $d = 128$. For each modality $m$, a dedicated linear projection layer maps the modality embedding $E_m$ into latent vector $z_m$:
$$z_m = W_m E_m + b_m, \quad W_m \in \mathbb{R}^{128 \times D_m}, \quad b_m \in \mathbb{R}^{128}$$
Projection weights are initialized orthogonally to preserve distance metrics across subspaces.

```
       [URL: 64-D]    [Text: 768-D]    [Image: 64-D]    [Audio: 64-D]    [Video: 64-D]
            |               |                |                |                |
         (W_url)         (W_txt)          (W_img)          (W_aud)          (W_vid)
            v               v                v                v                v
         [z_url]         [z_txt]          [z_img]          [z_aud]          [z_vid]
            +---------------+----------------+----------------+----------------+
                                             |
                                             v
                      [All Projected into 128-D Shared Space]
                                             |
                       +---------------------+---------------------+
                       | Pairwise Cosine Similarity:               |
                       | S_ij = (z_i . z_j) / (||z_i|| * ||z_j||)  |
                       +---------------------+---------------------+
                                             |
                                             v
                       +-------------------------------------------+
                       | Agreement Multiplier:                     |
                       | alpha = 0.8 + 0.4 * Mean(S_ij)            |
                       +-------------------------------------------+
```
*Fig. 2. Shared Latent Space projection and cosine agreement pipeline.*

### B. Pairwise Cosine Similarity & Agreement Multiplier
Given $K$ active modalities present on a webpage ($K \ge 2$), we calculate the pairwise cosine similarity for all pairs $(i, j)$:
$$S_{i, j} = \frac{z_i \cdot z_j}{\|z_i\|_2 \|z_j\|_2}$$
The mean cross-modal agreement is defined as:
$$\bar{S} = \frac{2}{K(K-1)} \sum_{i=1}^{K-1} \sum_{j=i+1}^{K} |S_{i, j}|$$

We map $\bar{S} \in [0, 1]$ into a dynamic **Agreement Multiplier** $\alpha$:
$$\alpha = 0.8 + 0.4 \times \bar{S}, \quad \alpha \in [0.8, 1.2]$$
* **Corroborated Attack:** When modalities exhibit structural congruence (e.g., high risk in both URL and text embeddings), $\bar{S} \to 1.0$, yielding $\alpha \to 1.2$ (+20% risk amplification).
* **Asymmetric Signal Conflict:** When an isolated media anomaly appears on an otherwise verified domain (e.g., standard entertainment video on YouTube), $\bar{S} \to 0.0$, yielding $\alpha \to 0.8$ (-20% risk dampening), mathematically preventing false-positive site blocks.

### C. Dynamic Confidence Weighting
Each modality possesses an intrinsic base importance weight ($w_{\text{url}}=1.5$, $w_{\text{text}}=1.2$, $w_{\text{image}}=1.0$, $w_{\text{audio}}=1.0$, $w_{\text{video}}=1.1$). We scale these dynamically by prediction confidence $c_m$:
$$c_m = 2 \times |R_m - 0.5|$$
where $R_m \in [0, 1]$ is the individual modality risk score. Highly polarizing predictions ($R_m \approx 1.0$ or $R_m \approx 0.0$) yield $c_m \approx 1.0$, whereas ambiguous predictions ($R_m \approx 0.5$) scale toward $c_m \approx 0.0$.

The base fusion score $S_{\text{base}}$ is computed as:
$$S_{\text{base}} = \frac{\sum_{m \in \mathcal{M}} w_m \cdot c_m \cdot R_m}{\sum_{m \in \mathcal{M}} w_m \cdot c_m}$$

The final aggregated risk score $R_{\text{final}}$ is formulated as:
$$R_{\text{final}} = \text{clip}(S_{\text{base}} \times \alpha, \, 0.01, \, 0.99)$$

### D. Threat Categorization and Mitigation Protocols
$R_{\text{final}}$ maps directly into five operational protocol states:
* $R_{\text{final}} \ge 0.80$: **🔴 CRITICAL** — Coordinated attack. Browser badge displays `!!`; DOM injection renders an unclosable full-screen warning overlay blocking user input.
* $0.60 \le R_{\text{final}} < 0.80$: **🟠 HIGH RISK** — Strong threat indicators. Injects warning overlay with bypass confirmation button.
* $0.40 \le R_{\text{final}} < 0.60$: **🟡 MEDIUM RISK** — Inconclusive/mixed signals. Browser badge displays `~`; cautionary notification advising verification.
* $0.20 \le R_{\text{final}} < 0.40$: **🔵 LOW RISK** — Minimal threat indicators; passive standard browsing.
* $R_{\text{final}} < 0.20$: **🟢 SAFE** — Verified legitimate architecture; badge displays `✓`.

---

## VI. Explainability Framework

To bridge the gap between machine learning predictions and end-user trust, PhishGuard incorporates a two-tiered explainability pipeline:

```
+-------------------------------------------------------------------------+
|                  EXPLAINABLE AI (XAI) PIPELINE                         |
|                                                                         |
|  [Extracted Features]                                                   |
|          |                                                              |
|          v                                                              |
|  +-------------------------------------------------------------------+  |
|  | Tier 1: Quantified SHAP Directional Impacts                       |  |
|  | - "Targeted Brand Impersonation"              : +0.35 [Risk Up]   |  |
|  | - "High-Risk Domain TLD (.tk)"                : +0.28 [Risk Up]   |  |
|  | - "Credential Harvesting Terminology"         : +0.22 [Risk Up]   |  |
|  | - "Known Trusted Domain (google.com)"         : -0.40 [Risk Down] |  |
|  +-------------------------------------------------------------------+  |
|          |                                                              |
|          v                                                              |
|  +-------------------------------------------------------------------+  |
|  | Tier 2: LLM Contextual Reasoning (Gemini 1.5 Pro)                 |  |
|  | Synthesizes mathematical SHAP vectors and raw page context into   |  |
|  | human-readable guidance:                                           |  |
|  | "HIGH RISK: Although hosted over HTTPS, this page imitates PayPal |  |
|  | on an unverified TLD (.tk) and requests an immediate OTP.         |  |
|  | Do NOT submit credentials."                                       |  |
|  +-------------------------------------------------------------------+  |
|          |                                                              |
|          v                                                              |
|  Rendered on User Warning Overlay & Extension Simulation Dashboard      |
+-------------------------------------------------------------------------+
```
*Fig. 3. Two-tiered explainability pipeline utilizing SHAP and LLM contextual synthesis.*

### A. Tier 1: Quantified SHAP Directional Feature Attribution
For tabular and structural URL inputs, the server computes directional impact values $\phi_i$ derived from Shapley additive explanations:
$$f(x) = \phi_0 + \sum_{i=1}^{P} \phi_i$$
Positive attributions ($\phi_i > 0$) highlight threat accelerants (e.g., `Brand Impersonation: +0.35`, `High-Risk TLD: +0.28`), while negative attributions ($\phi_i < 0$) highlight mitigating factors (e.g., `Known Trusted Domain: -0.40`, `Valid HTTPS: -0.05`). The top 4 ranked impacts are delivered to the extension UI.

### B. Tier 2: LLM Natural Language Contextual Synthesis
The feature dictionary, DOM context, and SHAP vectors are passed to **Google Gemini 1.5 Pro** via structured JSON prompts. The model translates mathematical vectors into precise, non-technical natural language instructions, resolving alert ambiguity for everyday users.

---

## VII. Experimental Evaluation and Results

### A. Datasets and Experimental Setup
Models were trained and evaluated on representative benign and malicious corpora:
* **URL Corpus:** 45,000 URLs (25,000 active phishing URLs harvested from PhishTank and OpenPhish; 20,000 benign URLs from Tranco Top 1M and authoritative Indian portals). Split: 85% train ($N=38,250$), 15% test ($N=6,750$).
* **Text Corpus:** 6,000 email/web samples (3,000 phishing samples from Nazario and SpamAssassin; 3,000 legitimate samples from Enron and EasyHam). Split: 85% train ($N=5,100$), 15% test ($N=900$).
* **Media Benchmarks:** ResNet-18 visual embeddings evaluated on cloned banking screenshots; audio evaluated on ASVspoof 2019/2021; video evaluated on Celeb-DF-v2.

### B. Single-Modality Classifier Performance
Table I reports the quantitative performance of individual detection modules on held-out test splits.

**TABLE I: Performance Metrics Across Individual Detection Modules**

| Detection Module | Architecture / Algorithm | Test Size | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **URL Detector** | RF + GB + LR Ensemble (57 feats) | 6,750 | **98.73%** | **99.06%** | **98.64%** | **0.9885** | **1.0000** |
| **Text NLP** | TF-IDF (1,695 tokens) + Logistic Reg. | 900 | **98.56%** | **97.60%** | **99.56%** | **0.9857** | **1.0000** |
| **Visual Anomaly** | Pretrained ResNet-18 + Isolation Forest | 400 | **94.20%** | **93.10%** | **95.40%** | **0.9424** | **0.9610** |
| **Audio Vishing** | ASVspoof Spectral + Isolation Forest | 600 | **93.80%** | **92.50%** | **94.10%** | **0.9329** | **0.9540** |
| **Video Deepfake**| Optical Flow Variance + Isolation Forest | 350 | **92.40%** | **91.80%** | **93.20%** | **0.9249** | **0.9480** |

The URL ensemble achieves exceptional discriminative power, while the text NLP module achieves 99.56% recall, ensuring nearly all coercive linguistic traps are flagged.

### C. Multimodal Fusion and False-Positive Reduction
To evaluate the impact of correlation-aware fusion, we benchmarked the complete multimodal pipeline against a dedicated unseen dataset (`unseen_test_data.csv` and `multimodal_test_data.csv`).

**TABLE II: Benchmark Comparison: Single vs. Multimodal Fusion**

| Configuration | Accuracy | Precision | Recall | F1-Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| URL Ensemble Only | 98.73% | 99.06% | 98.64% | 0.9885 | 1.12% |
| Text NLP Only | 98.56% | 97.60% | 99.56% | 0.9857 | 2.45% |
| Naive Score Averaging ($\sum \frac{R_i}{M}$) | 96.10% | 94.80% | 97.30% | 0.9603 | 4.80% |
| **PhishGuard Latent Fusion ($\alpha$-Weighted)** | **99.12%** | **99.20%** | **98.90%** | **0.9905** | **0.69%** |

As demonstrated in Table II, naive arithmetic score averaging increases false positives to 4.80% due to uncalibrated noise across modalities. In contrast, our correlation-aware latent fusion achieves **99.12% accuracy** and reduces the false-positive rate from 1.12% down to **0.69%**, representing a **38.4% reduction in false alarms** over the URL detector alone.

### D. Case Study: Cross-Modal Conflict on Video Platforms
To empirically validate our agreement multiplier under real browsing conditions, we tested the live extension against YouTube media streams:
1. **Case A (Genuine Media on Verified Domain):** Visiting Tom Cruise’s honorary Oscar speech (`youtube.com/watch?v=NWZKuueumJE`). The video detector identifies a video stream, but platform-aware heuristics recognize `youtube.com` as an authoritative media host. Text and URL risks evaluate to 3% and 7%. The fusion agreement drops to $\bar{S}=0.12$ ($\alpha = 0.831 \downarrow \text{Conflict}$). Aggregated score is **10% (🟢 SAFE)**. No disruptive warning is triggered.
2. **Case B (Phishing Video Lure on Lookalike Domain):** Visiting an urgent executive video impersonation hosted on a cloned domain (`urgent-video-login-75.html`). URL risk evaluates to 88%, text risk to 78%, and video anomaly to 72%. Embeddings align closely ($\bar{S}=0.85$, $\alpha = 1.200 \uparrow \text{Agreement}$). Aggregated risk escalates to **89% (🔴 CRITICAL)**, instantly triggering the full-screen warning overlay.

### E. Latency and Resource Utilization
Latency profiling was conducted over 250 browsing sessions connected to the live Render cloud microservice:
* Client DOM Vector Extraction: $14.2 \pm 3.1\text{ ms}$
* Network Round-Trip (TLS/HTTPS): $142.5 \pm 22.0\text{ ms}$
* Cloud Model Inference & Fusion: $6.8 \pm 1.4\text{ ms}$
* Total End-to-End Latency: **$163.5 \pm 24.2\text{ ms}$**

Total execution overhead remains well below the human perceptual threshold for web page rendering delay ($<250\text{ ms}$).

---

## VIII. Applicability in Other Domains

The mathematical framework of correlation-aware latent space projection is readily translatable to other adversarial security settings:

### A. Telemedicine and Digital Healthcare Prescription Fraud
* **Problem:** Fraudulent actors establish unlicensed online consultation portals, forging physician digital stamps and utilizing real-time synthetic deepfake video/audio to acquire controlled pharmaceutical prescriptions.
* **Methodology Mapping:**
  * *URL Module* $\to$ Telehealth registrar certificate and prescription QR cryptographic verification.
  * *Text NLP Module* $\to$ Clinical consistency inspection between diagnostic symptom notes and requested ICD-10/NDC drug dosages.
  * *Vision Module* $\to$ Convolutional verification of official medical board registration seals and institutional hospital stamps.
  * *Audio/Video Module* $\to$ Real-time facial artifact inspection during remote video consultation.
  * *Fusion Engine* $\to$ Replaces brand dictionaries with medical ontology registries (MIMIC-III), cross-verifying biometric and document authenticity.

### B. Remote KYC and Financial Loan Verification
Financial institutions conduct digital Know-Your-Customer (KYC) onboarding. Attackers deploy synthetic selfie videos (video injection attacks) paired with stolen PAN/Aadhaar identity credentials. PhishGuard’s cross-modal correlation multiplier provides a robust defense by mathematically testing whether document OCR text, facial liveness vectors, and application metadata mutually corroborate.

---

## IX. Conclusion and Future Work

In this paper, we presented **PhishGuard AI**, a comprehensive, real-time multimodal threat detection browser extension. By decoupling client-side DOM vectorization from cloud-hosted inference, PhishGuard achieves sub-200ms real-time browsing protection. Our 5-modality architecture addresses structural URL patterns, text semantics, visual layout, audio vishing, and deepfake video streams. Through our proposed Shared Latent Space Projection and pairwise cosine agreement multiplier, the system mathematically mitigates single-modality deception, achieving **99.12% multimodal accuracy** and decreasing false-positive alarms by **38.4%**. Furthermore, grounded SHAP attribution and LLM natural-language synthesis demystify threat predictions for everyday users.

Future research will focus on:
1. Exporting lightweight distilled models to WebAssembly/ONNX for native offline browser inference.
2. Expanding cross-browser support to Mozilla Firefox and Microsoft Edge.
3. Implementing decentralized, privacy-preserving federated learning pipelines to ingest emerging threat vectors continuously from open threat feeds.

---

### Acknowledgment
The authors express sincere gratitude to project guide **Dr. Roopa Ravish**, Professor, Department of Computer Science and Engineering, PES University, Bengaluru, for invaluable guidance and review throughout the development of Capstone Project `PW26_RR_06`.

---

### References

1. A. Aleroud and L. Zhou, "Phishing environments, techniques, and countermeasures: A survey," *Computers & Security*, vol. 68, pp. 160–196, 2017.
2. A. K. Jain and B. B. Gupta, "A machine learning based approach for phishing detection using hyperlinks information," *Journal of Ambient Intelligence and Humanized Computing*, vol. 10, no. 5, pp. 2015–2028, 2019.
3. N. Das et al., "Deepfake detection: A comprehensive review of methods, datasets, and challenges," *IEEE Access*, vol. 9, pp. 152173–152190, 2021.
4. S. Marchal, K. Saari, N. Singh, and N. Asokan, "Know your phishing: Diversity of phishing campaigns and their countermeasures," *IEEE Security & Privacy*, vol. 14, no. 5, pp. 44–54, 2016.
5. P. Shirazi, et al., "XF-PhishBERT: Explainable few-shot learning with modern BERT for emerging phishing attacks using XF PhishBERT," *Scientific Reports*, vol. 14, art. no. 12450, 2024.
6. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, vol. 30, pp. 4765–4774, 2017.
7. A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, vol. 30, pp. 5998–6008, 2017.
8. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.
9. Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, "Celeb-DF: A large-scale challenging dataset for deepfake forensics," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 3207–3216.
10. M. Todisco, et al., "ASVspoof 2019: Future horizons in spoofed and fake audio detection," *Interspeech 2019*, pp. 1008–1012, 2019.
