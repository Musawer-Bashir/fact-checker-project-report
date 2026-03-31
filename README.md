# GeoFact — Explainable AI Fact-Checking System
## University of West London - BSc Computer Science Final Year Project
### Musawer Bashir (21582512) | Supervisor: Ali Gheitasy

---

## Project Structure

```
factchecker/
├── 1_build_dataset.py     # Downloads FEVER + LIAR, filters, saves dataset.csv
├── 2_train_model.py       # Fine-tunes BERT, saves model, generates metrics
├── 3_shap_analysis.py     # Generates SHAP explanations + case studies
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── data/                  # Created automatically
│   ├── dataset.csv        # Your filtered training data
│   └── test_set.csv       # Held-out test set
├── model/
│   └── bert_factchecker/  # Saved fine-tuned BERT model
├── results/
│   ├── metrics.json       # Accuracy, F1, precision, recall
│   ├── baselines.json     # Comparison against baselines
│   ├── confusion_matrix.png
│   ├── case_studies.json
│   └── shap_figures/      # HTML SHAP plots for each case study
└── templates/
    ├── index.html         # Main UI
    └── about.html         # About page
```

---

## Setup (do this ONCE)

### 1. Requirements
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- GPU optional but speeds up training 10x (Google Colab works great)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create data folder
```bash
mkdir -p data results model
```

---

## Running the System (follow in order)

### Step 1: Build the dataset
```bash
python 1_build_dataset.py
```
- Downloads FEVER (~1GB) and LIAR datasets automatically from HuggingFace
- Filters for USA-China geopolitical claims
- Saves balanced binary dataset to data/dataset.csv
- Takes ~5-10 minutes

### Step 2: Train the BERT model
```bash
python 2_train_model.py
```
- Fine-tunes bert-base-uncased for 5 epochs
- Saves best model to model/bert_factchecker/
- Saves metrics.json and baselines.json
- Takes ~15-30 min with GPU, ~2-3 hours without
- TIP: Use Google Colab (free GPU) if training is slow

### Step 3: Generate SHAP explanations
```bash
python 3_shap_analysis.py
```
- Runs your 5 case studies
- Saves SHAP HTML figures to results/shap_figures/
- Open each .html file in your browser and screenshot for your report

### Step 4: Run the Flask app
```bash
python app.py
```
- Opens at http://localhost:5000
- Submit claims via the web interface
- Screenshot for your report figures

---

## Using Google Colab (Recommended if no GPU)

1. Go to https://colab.research.google.com
2. New notebook → Runtime → Change runtime type → GPU (T4)
3. Upload this entire folder or paste code cell by cell
4. Run in order: 1_, 2_, 3_, then app.py
5. For the Flask app in Colab, add these lines before app.run():
   ```python
   from pyngrok import ngrok
   public_url = ngrok.connect(5000)
   print(public_url)
   ```
   Then install: !pip install pyngrok

---

## For Your Report

### Section 4.2 — Dataset Statistics
After running Step 1, open data/dataset.csv and report:
- Total claims
- True vs False breakdown  
- Source breakdown (FEVER vs LIAR)

### Section 5.1 — Classification Results
After running Step 2, open results/metrics.json and results/baselines.json.
These contain all the numbers for your Table 3.

### Section 5.2 — SHAP Case Studies
After running Step 3, open each file in results/shap_figures/ in your browser.
Screenshot each one for Figures 11-15 in your report.
The analysis notes in 3_shap_analysis.py guide what to discuss for each.

### Section 4.5 — Flask Prototype Screenshots
Take 3 screenshots while app.py is running:
- Figure 7: The home page (before submission)
- Figure 8: A results page with verdict + SHAP visualisation
- Figure 9: The about page

---

## Common Issues

**"Out of memory" during training:**
Reduce batch size in 2_train_model.py:
```python
per_device_train_batch_size=8,  # change from 16 to 8
per_device_eval_batch_size=16,
```

**SHAP is very slow:**
SHAP with BERT is computationally expensive. For the case studies,
SHAP runs on 5 claims which takes ~2-5 minutes per claim on CPU.
Use GPU or reduce the number of SHAP perturbation samples:
```python
explainer = shap.Explainer(pipe, masker=shap.maskers.Text(tokenizer),
                           algorithm="partition", max_evals=100)
```

**Dataset is very small after filtering:**
The keyword filter may produce few matches. You can:
1. Add more keywords to the KEYWORDS list in 1_build_dataset.py
2. Use the full LIAR dataset without filtering (broader political dataset)
3. The model still works with 200+ claims; 500+ is better

**Flask can't find templates:**
Make sure you're running app.py from the factchecker/ directory,
not from a parent directory.

---

## References
- Devlin et al. (2019) BERT: Pre-training of Deep Bidirectional Transformers
- Thorne et al. (2018) FEVER: A Large-Scale Dataset for Fact Extraction and Verification
- Lundberg & Lee (2017) A Unified Approach to Interpreting Model Predictions (SHAP)
- Wang (2017) "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection
