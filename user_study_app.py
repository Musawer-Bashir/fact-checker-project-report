"""
User study Flask app for RQ2 evaluation.
Run: python3 user_study_app.py
Visit: http://localhost:5050
"""
import os
import csv
import random
from datetime import datetime
from flask import Flask, render_template_string, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = os.urandom(24)

TRIALS = [
    {"id": 1, "claim": "The COVID-19 virus originated from the Wuhan Institute of Virology in China.",
     "verdict": "False", "confidence": 55.6,
     "shap_html_path": "results/shap_figures/case_study_1.html"},
    {"id": 2, "claim": "China constructed artificial islands in the South China Sea featuring military airstrips.",
     "verdict": "True", "confidence": 65.7,
     "shap_html_path": "results/shap_figures/case_study_9.html"},
    {"id": 3, "claim": "The United States imposed 25% tariffs on $200 billion worth of Chinese goods during the trade war.",
     "verdict": "True", "confidence": 56.4,
     "shap_html_path": "results/shap_figures/case_study_3.html"},
    {"id": 4, "claim": "Hong Kong was returned to Chinese sovereignty in 1997 under a 'one country, two systems' framework.",
     "verdict": "True", "confidence": 80.7,
     "shap_html_path": "results/shap_figures/case_study_7.html"},
    {"id": 5, "claim": "The US government banned Huawei equipment citing national security risks related to Chinese espionage.",
     "verdict": "False", "confidence": 73.6,
     "shap_html_path": "results/shap_figures/case_study_5.html"},
    {"id": 6, "claim": "The 2022 Beijing Winter Olympics faced diplomatic boycotts from the US, UK and Canada.",
     "verdict": "False", "confidence": 64.3,
     "shap_html_path": "results/shap_figures/case_study_11.html"},
]

CSV_PATH = "user_study_results.csv"
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "participant_id", "trial_idx", "trial_id",
                    "condition", "trust", "comprehension", "willing_to_act",
                    "free_text"])

INFO_PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>Fact-Checking Study</title>
<style>body{font-family:system-ui;max-width:720px;margin:40px auto;padding:0 20px;line-height:1.6}
h1{color:#1a3a5c}.btn{background:#1a3a5c;color:#fff;padding:10px 24px;border:0;border-radius:4px;font-size:16px;cursor:pointer}</style></head>
<body><h1>Information and Consent</h1>
<p>You are invited to participate in a study evaluating an automated fact-checking system for USA-China geopolitical claims.</p>
<p><strong>What you will do:</strong> view 6 short claims and the system's verdict on each, then rate three aspects on a 1-7 scale. Some trials show an explanation highlighting which words the system used; others do not. Total time: about 10 minutes.</p>
<p><strong>Data:</strong> only your responses (no name, no email) are stored, anonymously, on a local file. The data will be reported in aggregate in a final-year project at the University of West London.</p>
<p><strong>Withdrawal:</strong> close the browser tab at any time and no data is saved.</p>
<p>By clicking Start, you consent to participate.</p>
<form method="post" action="/start"><button class="btn" type="submit">Start</button></form>
</body></html>
"""

TRIAL_PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>Trial {{ trial_idx + 1 }}/6</title>
<style>body{font-family:system-ui;max-width:780px;margin:30px auto;padding:0 20px;line-height:1.5}
.claim{background:#f4f7fa;border-left:4px solid #1a3a5c;padding:14px 18px;margin:14px 0;font-size:18px}
.verdict{display:inline-block;padding:6px 14px;border-radius:4px;font-weight:bold}
.true{background:#d4edda;color:#155724}.false{background:#f8d7da;color:#721c24}
.shap{border:1px solid #ddd;padding:10px;margin:14px 0;background:#fff}
fieldset{border:1px solid #ccc;margin:14px 0;padding:10px}
.likert{display:flex;justify-content:space-between;gap:6px}
.likert label{flex:1;text-align:center;font-size:13px;cursor:pointer}
.btn{background:#1a3a5c;color:#fff;padding:10px 24px;border:0;border-radius:4px;font-size:16px;cursor:pointer}
small{color:#666}</style></head>
<body>
<h2>Trial {{ trial_idx + 1 }} of 6</h2>
<div class="claim">{{ trial.claim }}</div>
<p><strong>System verdict:</strong>
<span class="verdict {{ 'true' if trial.verdict == 'True' else 'false' }}">{{ trial.verdict }}</span>
({{ trial.confidence }}% confidence)</p>
{% if condition == 'B' and shap_html %}
<p><strong>Explanation (highlighted words drove the verdict):</strong></p>
<div class="shap">{{ shap_html | safe }}</div>
{% endif %}
<form method="post" action="/answer">
<fieldset><legend>How much do you trust this verdict?</legend>
<div class="likert">{% for n in range(1,8) %}<label><input type="radio" name="trust" value="{{n}}" required>{{n}}<br><small>{{ ['Not at all','','','Neutral','','','Completely'][n-1] }}</small></label>{% endfor %}</div></fieldset>
<fieldset><legend>How well do you understand why the system made this verdict?</legend>
<div class="likert">{% for n in range(1,8) %}<label><input type="radio" name="comprehension" value="{{n}}" required>{{n}}<br><small>{{ ['Not at all','','','Neutral','','','Completely'][n-1] }}</small></label>{% endfor %}</div></fieldset>
<fieldset><legend>Would you act on this verdict (e.g. share, cite, repeat)?</legend>
<div class="likert">{% for n in range(1,8) %}<label><input type="radio" name="action" value="{{n}}" required>{{n}}<br><small>{{ ['Definitely not','','','Maybe','','','Definitely'][n-1] }}</small></label>{% endfor %}</div></fieldset>
<p>Optional comment: <input type="text" name="free_text" style="width:100%"></p>
<button class="btn" type="submit">Submit and continue</button>
</form></body></html>
"""

DONE_PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>Thank you</title>
<style>body{font-family:system-ui;max-width:680px;margin:80px auto;padding:0 20px;text-align:center}</style></head>
<body><h1>Thank you</h1><p>Your responses have been saved. You may now close this tab.</p></body></html>
"""


@app.route("/")
def index():
    return INFO_PAGE


@app.route("/start", methods=["POST"])
def start():
    pid = datetime.now().strftime("%Y%m%d%H%M%S") + str(random.randint(100, 999))
    session["pid"] = pid
    order = ["A", "B"] * 3 if random.random() < 0.5 else ["B", "A"] * 3
    random.shuffle(order)
    session["conditions"] = order
    trial_order = list(range(len(TRIALS)))
    random.shuffle(trial_order)
    session["trial_order"] = trial_order
    session["trial_idx"] = 0
    return redirect(url_for("trial"))


@app.route("/trial")
def trial():
    idx = session.get("trial_idx", 0)
    if idx >= len(TRIALS):
        return redirect(url_for("done"))
    trial_id = session["trial_order"][idx]
    cond = session["conditions"][idx]
    trial = TRIALS[trial_id]

    shap_html = ""
    if cond == "B" and os.path.exists(trial["shap_html_path"]):
        with open(trial["shap_html_path"]) as f:
            shap_html = f.read()

    return render_template_string(TRIAL_PAGE, trial=trial, trial_idx=idx,
                                  condition=cond, shap_html=shap_html)


@app.route("/answer", methods=["POST"])
def answer():
    idx = session["trial_idx"]
    trial_id = session["trial_order"][idx]
    cond = session["conditions"][idx]
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.now().isoformat(),
                    session["pid"], idx, trial_id, cond,
                    request.form.get("trust"),
                    request.form.get("comprehension"),
                    request.form.get("action"),
                    request.form.get("free_text", "")])
    session["trial_idx"] = idx + 1
    return redirect(url_for("trial"))


@app.route("/done")
def done():
    return DONE_PAGE


if __name__ == "__main__":
    print("\nUser study app running at http://localhost:5050")
    print(f"Results will accumulate in {CSV_PATH}\n")
    app.run(host="0.0.0.0", port=5050, debug=False)