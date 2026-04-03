# GeoFact — Explainable AI Fact-Checking System

BSc Computer Science Final Year Project — University of West London
Student: Musawer Bashir (21582512) | Supervisor: Ali Gheitasy

## Overview
GeoFact classifies USA-China geopolitical claims as True or False using a fine-tuned BERT model, with SHAP token-level explanations delivered through a Flask web interface.

## Results
- BERT macro F1: 70.3% on held-out test set
- TF-IDF baseline: 72.4%
- Dataset: 454 labelled USA-China geopolitical claims
- SHAP sufficiency analysis conducted across 5 case studies

## How to Run
1. pip install -r requirements.txt
2. python 1_build_dataset.py
3. python 2_train_model.py
4. python 3_shap_analysis.py
5. python app.py  →  open http://localhost:5001

## Note
Model weights and dataset excluded due to file size. Run steps 1-3 to reproduce.
