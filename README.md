# HEPID — Healthcare Explainable Prompt Injection Defense

## Overview
An Explainable Multi-Layer Framework for Detecting and Mitigating 
Indirect Prompt Injection Attacks in Healthcare RAG Systems.

**M.Tech Major Project**  
Amrita Vishwa Vidyapeetham, Amritapuri Campus  
Amrita Center for Cybersecurity Systems and Networks  

**Author:** Alphin Kayalathu Mathew (AM.SC.P2CSN24002)  
**Guide:** Devi Rajeev  
**Co-Guide:** Akshara Ravi  

---

## Key Results

| Metric | Value |
|--------|-------|
| AUC-ROC Risk Fusion | 0.9087 |
| AUC-ROC Baseline | 0.8224 |
| Precision | 93.66% |
| Recall | 98.44% |
| F1 Score | 75.88% |
| Suspicious chunks defused | 89.1% |
| Avg inference time | 14.26 ms |

---

## Risk Score Formula

risk = 0.7 x DistilBERT_probability + 0.3 x threat_indicator_score

## Three-Tier Routing

- Benign risk less than 0.50 — Direct pass  
- Suspicious 0.50 to 0.80 — LIME sanitization and re-score  
- Malicious risk greater than 0.80 — Hard block  

---

## Project Structure

- app.py — Streamlit demo UI  
- detector.py — Core HEPID pipeline  
- evaluate.py — Three-layer evaluation  
- rag.py — RAG pipeline with FAISS  
- simple_train.py — DistilBERT fine-tuning  
- metrics_report.json — Evaluation results  
- rag_index.pkl — Pre-built FAISS index  
- medical_db — 8 medical PDFs  
- models — Training and comparison scripts  
- scripts — Dataset creation scripts  

---

## Setup

pip install -r requirements.txt

streamlit run app.py

python evaluate.py

---

## Dataset

- Base: MedQuAD Medical Question Answering Dataset  
- Training: crafted_instruction_data_medquad.json  
- Evaluation: external_test_dataset.json 900 samples 50/50 balanced  

---

## Tech Stack

- Detection: DistilBERT fine-tuned  
- Explainability: LIME  
- Retrieval: FAISS with sentence-transformers all-MiniLM-L6-v2  
- LLM: Gemini 2.5 Flash  
- UI: Streamlit  

---

## References

- Greshake et al. 2023. Not What You Signed Up For  
- OWASP Top 10 for LLM Applications LLM01 Prompt Injection  
- MITRE ATLAS ML Attack Techniques  
- Perez and Ribeiro 2022. Ignore Previous Prompt  
