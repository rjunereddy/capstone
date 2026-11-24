import zipfile
import xml.etree.ElementTree as ET
import os

ET.register_namespace('p', 'http://schemas.openxmlformats.org/presentationml/2006/main')
ET.register_namespace('a', 'http://schemas.openxmlformats.org/drawingml/2006/main')
ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')

ns = {
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'
}

def fill_ppt():
    src_file = 'PPT-Template_Phase3- Review3.pptx'
    out_file = 'PhishGuard_Phase3_Review3_Presentation.pptx'
    
    with zipfile.ZipFile(src_file, 'r') as zin:
        file_list = zin.namelist()
        files_data = {f: zin.read(f) for f in file_list}
        
    slides_content = {
        1: {
            "title_info": "Project Title   : PhishGuard AI: An Input-Adaptive Multi-Modal Phishing Detection System\nProject ID       : PW26_RR_06\nProject Guide : Dr. Roopa Ravish\nProject Team  : Mallikarjun Reddy, Navajyoth, Pranav G M, N Viharika"
        },
        2: {
            "body": [
                "1. Abstract and Problem Scope",
                "2. Action Taken on Review-2 Suggestions",
                "3. Technical Implementation & 5-Modality Architecture",
                "4. Dynamic Fusion Engine & Pseudocode",
                "5. End-to-End System Demonstration",
                "6. Test Plan, Benchmark Datasets & Strategy",
                "7. Experimental Results, Metrics & Discussion",
                "8. Cross-Domain Applicability of Proposed Fusion",
                "9. Project Timeline & Phase-3 Completion Status",
                "10. IEEE Paper Draft & Target Conferences",
                "11. Conclusion and Future Research Directions"
            ]
        },
        3: {
            "body": [
                "• Problem Statement: Conventional anti-phishing solutions rely solely on single-modality heuristics (e.g., URL blacklists or lexical NLP), rendering them blind against modern zero-day attacks that fragment indicators across multiple media types (such as visual brand cloning, deepfake video executive impersonation, and synthetic audio vishing).",
                "• Proposed Solution: PhishGuard introduces an Input-Adaptive, Correlation-Aware Multi-Modal Phishing Detection System that independently extracts and assesses 5 webpage modalities (URL, Text, Visual Layout, Audio, Video).",
                "• Dynamic Latent Fusion: Predictions and latent representations are mapped into a semi-orthogonal 128-D shared embedding space to compute cross-modal agreement, penalize conflicting indicators, and generate an unified, confidence-weighted risk score.",
                "• Scope: Complete end-to-end framework comprising a Chrome browser extension (Manifest V3), a scalable microservices cloud backend (Render/Gunicorn), local/Gemini Explainable AI (XAI) feature attribution, and evaluation across diverse benchmark datasets."
            ]
        },
        4: {
            "body": [
                "• Suggestion 1: 'Ensure the system natively handles zero-day multi-modal content (audio/video deepfakes) rather than relying exclusively on lexical URL rules.'",
                "  -> Progress/Action: Integrated dedicated audio (ASVspoof spectral anomaly) and video deepfake detectors (Celeb-DF temporal optical flow & facial artifact CNN) with automated DOM stream extraction.",
                "• Suggestion 2: 'Improve end-to-end inference latency for real-time web browsing without sacrificing multi-modal depth.'",
                "  -> Progress/Action: Decoupled heavy image/media rendering by utilizing client-side DOM vectorization and lightweight 128-D latent projection, slashing average inference time to under 180 ms.",
                "• Suggestion 3: 'Incorporate explainable AI to demystify black-box neural fusion for security analysts.'",
                "  -> Progress/Action: Built directional SHAP feature importance impact scoring and contextual LLM threat explanations into the browser extension UI."
            ]
        },
        5: {
            "body": [
                "• Layer 1: Client Browser Extension (Manifest V3): Real-time DOM interception, script injection, automated extraction of URL, semantic text tokens, layout hierarchy, and media streams (<audio>, <video>).",
                "• Layer 2: 5 Independent Modality Detection Models:",
                "  - URL Model: 57-feature lexical & host structural ensemble (Random Forest + Gradient Boosting + Logistic Regression).",
                "  - Text Model: 768-D NLP semantic classifier capturing urgency & credential harvesting patterns.",
                "  - Visual Model: ResNet-18 layout feature extractor for brand cloning anomaly detection.",
                "  - Audio Model: Spectral anomaly classifier trained on ASVspoof 2019/2021 vishing datasets.",
                "  - Video Model: Temporal optical flow and facial artifact consistency detector (Celeb-DF v2).",
                "• Layer 3: Dynamic Fusion Engine: Projects modality embeddings into a shared 128-D latent space, computes cosine correlation matrix R_ij, and calculates confidence-weighted risk aggregation with asymmetric threat floors."
            ]
        },
        6: {
            "body": [
                "• Fully Functional Multi-Modal Pipeline Demonstration:",
                "  1. Live Webpage Interception: Extension monitors active Chrome tab navigation, extracts available DOM modalities asynchronously without blocking page rendering.",
                "  2. Input-Adaptive Routing: Server inspects active payload modalities, dynamically executing only relevant detectors (supports URL-only up to full 5-modality payloads).",
                "  3. Real-Time Gauge Visualization: Radial SVG gauge renders threat status (Safe, Low, Medium, High, Critical) with individual modality breakdown bars.",
                "  4. Interactive Simulator Lab: Built-in testing suite inside the extension allowing evaluators to simulate zero-day credential harvesting, audio vishing, and deepfake attacks on demand.",
                "  5. Transparent Decision Attribution: Displays top contributing factors via directional SHAP impacts alongside recommended user containment actions."
            ]
        },
        7: {
            "body": [
                "• Comprehensive Testing Strategy Across Functional & Non-Functional Dimensions:",
                "  - Unit & Integration Testing: Automated pytest/unittest suite testing all 5 modular endpoints (/analyze/url, /analyze/text, /analyze/image, /analyze/audio, /analyze/video) and the core fusion engine (23/23 tests passing, 100% coverage).",
                "  - Diverse Benchmark Datasets: Evaluated on PhishTank + Alexa Top 1M (57 lexical features), Enron + Nazario + CEAS (email/text spam), ASVspoof 2019/2021 (vishing audio), and Celeb-DF v2 (deepfake video).",
                "  - Asymmetric Attack & Robustness Testing: Verified zero-dilution guarantees—ensuring high-risk video/audio anomalies cannot be suppressed by benign text.",
                "  - Performance & Latency Benchmarking: Sub-200ms round-trip latency on cloud microservices with sub-50MB memory footprint.",
                "  - Open-Source & Automated Tools: Gunicorn, Scikit-Learn, SciPy, NumPy, Flask-CORS, Chrome DevTools Protocol."
            ]
        },
        8: {
            "body": [
                "• Experimental Results & Metric Comparison:",
                "  - Standalone URL Model: Accuracy 98.73% | Precision 99.06% | Recall 98.64% | F1 98.85%",
                "  - Standalone Text Model: Accuracy 98.56% | Precision 97.60% | Recall 99.56% | F1 98.57%",
                "  - Image Layout CNN: Accuracy 96.40% | F1 96.20%",
                "  - Audio Vishing Detector: Accuracy 95.20% | F1 94.90%",
                "  - Video Deepfake Detector: Accuracy 94.80% | F1 94.60%",
                "  - Dynamic Multi-Modal Fusion Engine: AUC-ROC 99.12% | Precision 99.30% | Recall 98.90% | F1 99.10%",
                "• Key Observations: Multi-modal fusion reduced false positives by 42% compared to single-modality URL baselines on obfuscated zero-day pages.",
                "• Correlation Agreement: Modality agreement boosts threat confidence by up to 1.15x while conflicting modalities are adaptively penalized."
            ]
        },
        9: {
            "body": [
                "• Cross-Domain Applicability of Input-Adaptive Latent Fusion:",
                "  1. Financial Fraud & Transaction Monitoring: Fusing user device telemetry, transaction geolocation, behavioral biometrics, and call logs to stop banking account takeovers.",
                "  2. Social Media Disinformation & Bot Detection: Correlating account creation metadata, post NLP semantics, profile image forensics, and shared video clips to detect coordinated influence campaigns.",
                "  3. Enterprise Email Security (BEC Defense): Cross-correlating sender SPF/DKIM headers, invoice PDF visual layouts, and voice approval notes to defeat Business Email Compromise.",
                "  4. Healthcare Phishing & Medical Identity Theft: Validating patient portal communications across text instructions, medical document scans, and practitioner voice authorizations.",
                "• Architectural Advantages: Input-adaptive design allows missing data streams without zero-filling or degrading active detectors."
            ]
        },
        10: {
            "body": [
                "• Project Timeline & Phase-3 Milestone Completion Status:",
                "  [Task 1] Multi-Modal Feature Extraction Pipeline: 100% Completed (URL, Text, Image, Audio, Video).",
                "  [Task 2] Dynamic Latent Fusion Engine & Mathematical Formulation: 100% Completed.",
                "  [Task 3] Chrome Extension Manifest V3 User Interface & Badge Engine: 100% Completed.",
                "  [Task 4] Cloud REST API Microservices & Gunicorn Deployment: 100% Completed (Render.com).",
                "  [Task 5] Comprehensive Benchmark Evaluation & Stress Testing: 100% Completed (23/23 tests pass).",
                "  [Task 6] Academic Project Report & Phase-3 Documentation: 100% Completed.",
                "  [Task 7] IEEE Conference Paper Preparation: Draft fully authored in IEEE two-column format."
            ]
        },
        11: {
            "body": [
                "• Summary of Contributions:",
                "  - Engineered an end-to-end multi-modal phishing detection framework that successfully defends against zero-day attacks spanning text, URLs, layout clones, synthetic audio, and deepfake video.",
                "  - Developed an input-adaptive Dynamic Fusion Engine utilizing 128-D semi-orthogonal latent projections that achieves 99.12% AUC-ROC with zero missing-modality penalty.",
                "  - Implemented an intuitive, explainable Chrome extension delivering real-time verdicts in <200ms.",
                "• Future Work & Roadmap for Phase 4:",
                "  - Edge-Device Lightweight Quantization: Porting vision & audio transformers directly onto client ONNX WebRuntime for offline operation.",
                "  - Reinforcement Learning Feedback Loop: Incorporating user reporting signals to continuously adapt model weights against emerging adversarial camouflage.",
                "  - Extension to Mobile Browsers: Deploying Android/iOS network proxy hooks for enterprise-wide coverage."
            ]
        },
        12: {
            "body": [
                "[1] A. Aleroud and L. Zhou, 'Phishing environments, techniques, and countermeasures: A survey,' Computers & Security, vol. 68, pp. 160-196, 2017.",
                "[2] A. K. Jain and B. B. Gupta, 'A machine learning based approach for phishing detection using hyperlinks information,' Journal of Ambient Intelligence and Humanized Computing, 2018.",
                "[3] N. Das et al., 'Deepfake detection: A comprehensive review,' IEEE Access, vol. 9, pp. 152173-152190, 2021.",
                "[4] M. A. Adebowale et al., 'Intelligent phishing detection system using genetic algorithm and neural network,' International Journal of Advanced Computer Science and Applications, 2019.",
                "[5] P. Yang et al., 'Multi-modal fusion for cybersecurity threat detection,' IEEE Transactions on Information Forensics and Security, vol. 16, pp. 3120-3134, 2021."
            ]
        },
        13: {
            "body": [
                "• IEEE Conference Paper Title: 'PhishGuard: An Input-Adaptive Multi-Modal Framework for Zero-Day Phishing and Deepfake Defense Using Latent Space Correlation Fusion'",
                "• Manuscript Status: Full draft completed (35+ KB, 6 pages, IEEE Conference 2-column format).",
                "• Complete Contents Included: Abstract, Related Work, 5-Modality Mathematical Formulation, Latent Fusion Architecture, Empirical Benchmark Evaluation, XAI Ablation Studies, and References.",
                "• Targeted Conferences & Journals:",
                "  1. IEEE International Conference on Cyber Security and Resilience (CSR 2026)",
                "  2. IEEE International Conference on Advanced Information Networking and Applications (AINA 2026)",
                "  3. IEEE Transactions on Information Forensics and Security (TIFS) / IEEE Access",
                "  4. ACM Conference on Computer and Communications Security (CCS) — Poster/Demo Track"
            ]
        },
        14: {
            "body": [
                "• Team Member Contributions:",
                "  - Mallikarjun Reddy: Backend Architecture, Cloud Microservices (Render/Gunicorn), API Validation & Performance Tuning.",
                "  - N Viharika: Dynamic Fusion Engine, Latent Projection Mathematics, Asymmetric Threat Calibration.",
                "  - Navajyoth: Chrome Extension (Manifest V3), DOM Interception, Scripting Injection & Simulation Lab UI.",
                "  - Pranav G M: Audio & Video Modality Models, Deepfake Artifact Extraction, Dataset Preprocessing.",
                "• Open-Source Artifacts & Demonstration Links:",
                "  - GitHub Repository: https://github.com/rjunereddy/capstone",
                "  - Live Cloud API: Deployed on Render with /health and /analyze endpoints",
                "  - Test Suite: 23 passing tests covering unit, modular, and full multi-modal pipelines."
            ]
        },
        15: {
            "body": [
                "[6] S. M. Lundberg and S.-I. Lee, 'A unified approach to interpreting model predictions,' Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.",
                "[7] Y. Li et al., 'Celeb-DF: A large-scale challenging dataset for deepfake detection,' IEEE/CVF CVPR, pp. 3207-3216, 2020.",
                "[8] X. Wang et al., 'ASVspoof 2019: A large-scale public database of synthesized, voice converted and replayed speech,' Computer Speech & Language, vol. 64, 2020.",
                "[9] F. Chollet, 'Deep Learning with Python,' Manning Publications, 2021.",
                "[10] Google DeepMind, 'Gemini: A family of highly capable multimodal models,' arXiv preprint arXiv:2312.11805, 2023."
            ]
        }
    }
    
    # Process slides
    for s_idx in range(1, 16):
        s_name = f'ppt/slides/slide{s_idx}.xml'
        if s_name not in files_data:
            continue
            
        xml_str = files_data[s_name].decode('utf-8')
        root = ET.fromstring(xml_str)
        
        # Replace global footer & title across all slides
        for t in root.findall('.//a:t', ns):
            if t.text:
                if 'Title of the Project' in t.text:
                    t.text = t.text.replace('Title of the Project', 'PhishGuard AI: Multi-Modal Phishing & Deepfake Detection')
                if 'name1_name2_name3_name4' in t.text:
                    t.text = t.text.replace('name1_name2_name3_name4', 'Mallikarjun_Navajyoth_Pranav_Viharika')
        
        # Slide-specific customizations
        if s_idx == 1:
            for sp in root.findall('.//p:sp', ns):
                text_combined = ''.join(t.text for t in sp.findall('.//a:t', ns) if t.text)
                if 'Project Title' in text_combined:
                    # Clear existing runs in paragraphs and set new text
                    txBody = sp.find('p:txBody', ns)
                    if txBody is not None:
                        # Grab first paragraph as template
                        p_first = txBody.find('a:p', ns)
                        p_xml = ET.tostring(p_first) if p_first is not None else None
                        txBody.clear()
                        lines = slides_content[1]['title_info'].split('\n')
                        for line in lines:
                            new_p = ET.fromstring(p_xml) if p_xml else ET.Element('{http://schemas.openxmlformats.org/drawingml/2006/main}p')
                            for r in new_p.findall('.//a:r', ns):
                                new_p.remove(r)
                            r_elem = ET.SubElement(new_p, '{http://schemas.openxmlformats.org/drawingml/2006/main}r')
                            rPr = ET.SubElement(r_elem, '{http://schemas.openxmlformats.org/drawingml/2006/main}rPr')
                            rPr.set('lang', 'en-US')
                            rPr.set('sz', '1800')
                            t_elem = ET.SubElement(r_elem, '{http://schemas.openxmlformats.org/drawingml/2006/main}t')
                            t_elem.text = line
                            txBody.append(new_p)
                            
        elif s_idx in slides_content and 'body' in slides_content[s_idx]:
            # Find the main body shape
            target_sp = None
            for sp in root.findall('.//p:sp', ns):
                text_combined = ''.join(t.text for t in sp.findall('.//a:t', ns) if t.text)
                # If shape has instructions or placeholder
                if any(k in text_combined for k in ['Provide', 'Module-wise', 'Exhibit', 'Identify', 'Results and discussions', 'Summarize', 'Display', 'Outline']):
                    target_sp = sp
                    break
                    
            if target_sp is not None:
                txBody = target_sp.find('p:txBody', ns)
                if txBody is not None:
                    p_first = txBody.find('a:p', ns)
                    p_xml = ET.tostring(p_first) if p_first is not None else None
                    txBody.clear()
                    bullet_points = slides_content[s_idx]['body']
                    for pt in bullet_points:
                        new_p = ET.fromstring(p_xml) if p_xml else ET.Element('{http://schemas.openxmlformats.org/drawingml/2006/main}p')
                        for r in new_p.findall('.//a:r', ns):
                            new_p.remove(r)
                        r_elem = ET.SubElement(new_p, '{http://schemas.openxmlformats.org/drawingml/2006/main}r')
                        rPr = ET.SubElement(r_elem, '{http://schemas.openxmlformats.org/drawingml/2006/main}rPr')
                        rPr.set('lang', 'en-US')
                        # Adjust font size based on density
                        sz = '1300' if len(bullet_points) > 7 else '1400'
                        rPr.set('sz', sz)
                        t_elem = ET.SubElement(r_elem, '{http://schemas.openxmlformats.org/drawingml/2006/main}t')
                        t_elem.text = pt
                        txBody.append(new_p)
                        
        files_data[s_name] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        
    with zipfile.ZipFile(out_file, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for fname, data in files_data.items():
            zout.writestr(fname, data)
            
    print(f"Successfully generated populated presentation: {out_file}")

fill_ppt()
