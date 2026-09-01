# PHISHGUARD: MULTI-MODAL PHISHING DETECTION SYSTEM

## Slide 1: Title Slide
- **Project Title**: PhishGuard: Multi-Modal Phishing Detection System
- **Project ID**: [Your Project ID]
- **Project Guide**: [Your Guide's Name]
- **Project Team**: [Your Names & SRNs]

---

## Slide 2: Outline
- Abstract and Scope
- Suggestions from Phase-2 Review 2
- Implementation Details (Technical)
- Results and Demo
- Individual Contribution
- Project Progress Plan (Phase 3 & 4)
- References

---

## Slide 3: Abstract and Scope
**Text to put on slide:**
- **Problem:** Single-modality phishing detection (e.g., just checking URLs) fails against modern, multi-layered attacks like deepfakes and visually deceptive clones.
- **Solution:** A Multi-Modal Phishing Detection System that mathematically fuses Text, URL, Visual (DOM), and Audio/Video analysis.
- **Scope:** Real-time web analysis via a Manifest V3 Browser Extension, backed by a Python/Flask Cloud Fusion Engine and Vertex AI Gemini for human-readable threat explanations.

*(INSERT IMAGE HERE: A professional, high-level infographic or icon set showing 4 inputs (Text, URL, Image, Video) combining into a single shield icon).* 

---

## Slide 4: Suggestions from Phase 2 - Review 2
**Text to put on slide:**
- **Suggestion 1:** "Ensure the system can detect deepfake video payloads natively, rather than just relying on text/URL heuristics."
  - **Feasibility/Progress:** Implemented structural DOM parsing to intercept `<video>` and `<audio>` tags. Transcoded demo videos to H.264 for seamless browser integration.
- **Suggestion 2:** "Improve latency; cloud processing takes too long for real-time browsing."
  - **Feasibility/Progress:** Shifted from cloud-rendering full screenshots to client-side DOM extraction. Average inference time reduced to <200ms on Render.com.

---

## Slide 5: Implementation Details (Methodology & Code)
**Text to put on slide:**
- **Frontend:** React-based UI & Manifest V3 Chrome Extension.
- **Backend:** Flask REST API deployed on Render.com.
- **Machine Learning Models:**
  - Text: DistilBERT (Transformer-based NLP).
  - URL: Ensemble statistical classifier.
  - Visual/Media: DOM-structure heuristics & CNNs.
- **Fusion Engine:** Computes Cosine Similarity between modality embeddings. Uses Dynamic Weighting to penalize conflicting anomalies and boost correlated threats, drastically reducing false positives.

*(INSERT IMAGE HERE: Paste a screenshot of your `phishfusion.py` code highlighting the mathematical logic, OR insert the System Architecture block diagram here).* 

---

## Slide 6: Results and Demo
**Text to put on slide:**
- Successfully intercepted deepfake media and credential harvesting clones that bypassed standard URL blacklists.
- **Performance:** System classifies threats in ~150-200ms post-spin-up.
- **Explainability:** Vertex AI successfully translates mathematical SHAP values into readable UI explanations.

*(INSERT IMAGE HERE: This is the most important slide. Put TWO screenshots here: 1. The Red Warning Overlay triggering on the fake PayPal/Deepfake page. 2. A screenshot of the extension's Live Analysis Tab showing the Risk Score Gauge).* 

---

## Slide 7: Individual Contribution
**Text to put on slide:**
- **[Name 1]:** Cloud Architecture (Render.com) & Flask API Deployment.
- **[Name 2]:** Fusion Engine Logic (Cosine Similarity & Dynamic Weighting).
- **[Name 3]:** Manifest V3 Browser Extension & DOM Extraction Scripts.
- **[Name 4]:** Vertex AI Gemini Integration & React Dashboard UI.
*(Note: Edit these to match your actual team roles).* 

---

## Slide 8: Project Progress Plan for Phase 3 and Phase 4
**Text to put on slide:**
- **Phase 3:** Mobile platform integration (iOS/Android) & continuous online learning pipelines.
- **Phase 4:** Enterprise dashboarding and threat-intelligence feed integration.

*(INSERT IMAGE HERE: Create a simple Gantt Chart in Excel or PowerPoint showing these tasks spanning across the upcoming months).* 

---

## Slide 9: References
**Text to put on slide:**
- [1] A. Aleroud and L. Zhou, "Phishing environments, techniques, and countermeasures: A survey," Computers & Security, vol. 68, pp. 160-196, 2017.
- [2] A. K. Jain and B. B. Gupta, "A machine learning based approach for phishing detection using hyperlinks information," Journal of Ambient Intelligence and Humanized Computing, 2018.
- [3] N. Das, et al., "Deepfake detection: A comprehensive review," IEEE Access, vol. 9, pp. 152173-152190, 2021.

---

## Slide 10: Any other information
**Text to put on slide:**
- **Live Demo:** We are prepared to demonstrate the real-time interception of a deepfake corporate announcement using our custom local server and cloud API.

*(INSERT IMAGE HERE: You can put a QR code here linking to your project's GitHub repository if you want to impress the panel).* 

---

## Slide 11: Thank You
**Text to put on slide:**
- Questions?
