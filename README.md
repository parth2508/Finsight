# FinSight — Personal Finance Analytics

Built this because I wanted to go beyond just plotting graphs — I wanted to actually detect something useful from data. Personal finance felt like a good starting point since everyone has transaction data and nobody really analyses it.

Live app: https://finsight-wckgnca8dzqfejluqwytu2.streamlit.app

---

## What it does

Three sections:

**Anomaly Detector** — flags transactions that look unusual compared to your normal spending. You can choose between Z-Score, IQR, or an ensemble of both. IQR works per category which makes it smarter — a Rs 2,000 Utilities bill is normal but a Rs 2,000 Food transaction probably isn't.

**Spending Insights** — shows where your money actually went. Category totals, daily trend, top spending days, monthly heatmap.

**Learn** — plain English explanation of how Z-Score and IQR work, with the actual formulas. Added this because I kept having to explain it to people who used the app.

---

## Tech stack

Python, pandas, numpy, scipy, seaborn, matplotlib, SQLite, Streamlit

---

## How to use

Three ways to input data — upload a CSV or Excel file, enter transactions manually one by one, or just load the sample data to see how it works.

If you're uploading, the file needs exactly three columns:

| date | category | amount |
|---|---|---|
| 2024-01-01 | Food | 450 |
| 2024-01-02 | Transport | 200 |

Date in YYYY-MM-DD format. Category must be one of: Food, Transport, Shopping, Utilities, Entertainment, Healthcare. Amount as a plain number — no Rs symbol or commas.

---

## How the detection works

**Z-Score** — measures how many standard deviations a transaction is from the mean. Anything beyond 2.5 gets flagged. Simple but treats all categories the same.

**IQR** — works per category. Calculates Q1, Q3, and the interquartile range, then flags anything outside 1.5x the fence. More accurate because it compares transactions within the same category.

**Ensemble** — only flags if both methods agree. Fewer false alarms but misses more anomalies.

Results on synthetic test data with 20 injected anomalies:

| Method | Precision | Recall |
|---|---|---|
| Z-Score | 100% | 45% |
| IQR | 73% | 95% |
| Ensemble | 100% | 45% |

IQR caught 19 out of 20 anomalies. Ensemble caught 9 but never wrongly flagged anything.

---

## Run it locally

```bash
git clone https://github.com/parth2508/Finsight.git
cd Finsight
pip install -r requirements.txt
streamlit run app.py
```

---

## Files
