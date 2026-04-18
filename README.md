# 🛒 GroceryGuard — Customer Churn Prediction for Instacart

**GroceryGuard** is an end-to-end machine learning system that identifies Instacart customers who are likely to stop ordering — before they actually leave. It flags every customer as HIGH, MEDIUM, or LOW risk and tells you exactly how many days you have left to reach them.

---

## The Problem

Every month, thousands of customers quietly stop ordering on Instacart. There is no goodbye, no complaint, no warning — they just disappear. By the time anyone notices, it is too late. The customer is gone and the only option is an expensive re-acquisition campaign to replace them.

This project builds a system to see the silence coming.

---

## What It Does

- Analyses the full order history of **206,209 customers** across **3 million transactions**
- Learns each customer's personal shopping rhythm — how often they order, how consistent they are, how loyal they are to certain products
- Flags customers showing signs of drift before they hit the churn threshold
- Assigns a risk label to every customer:

| Label | Meaning | Action |
|---|---|---|
| 🔴 HIGH RISK | Very likely to stop ordering soon | Act immediately |
| 🟡 MEDIUM RISK | Showing early warning signs | Monitor closely |
| 🟢 LOW RISK | Healthy, loyal customer | No action needed |

- For every HIGH RISK customer: provides a **personalised churn window** (how many days you have left to reach them) and a **recommended retention action** based on what they actually buy

---

## The Business Case

| Metric | Value |
|---|---|
| HIGH RISK customers flagged | 8,432 |
| Monthly revenue at risk | $1.26M |
| Annual revenue at risk | $15.1M |
| Monthly re-acquisition cost (if lost) | $792,608 |
| **Retaining 30% = monthly saving** | **$617,320** |
| **Annual saving** | **$7.4M** |
| Annual system running cost | ~$7,600 |
| **ROI** | **97,000%** |

> For every $1 spent running this system, it protects $970 in revenue.

---

## How It Works — Plain English

**Step 1 — Understanding each customer's normal**
Instead of comparing customers to each other, the system compares each customer to themselves. A weekly shopper going quiet for 3 weeks is a very different signal from a monthly shopper missing one order. The system understands that difference.

**Step 2 — Spotting the drift**
Two key signals drive the model:
- `median_days_between` — the customer's true ordering rhythm (uses median, not mean, so one long gap does not distort the picture)
- `gap_std` — how consistent that rhythm is. A sudden spike in inconsistency is often the first sign a customer is drifting away.

**Step 3 — Predicting before it happens**
The model is trained on historical order data *excluding* each customer's most recent order. This prevents data leakage — the model cannot "cheat" by seeing the answer during training. In real deployment, this means predictions are made while there is still time to act.

**Step 4 — Outputting something actionable**
The final output is not just a prediction. It is a retention list — every HIGH RISK customer paired with their department preference, their churn deadline, and a suggested action.

---

## Dataset

**Instacart Market Basket Analysis** — publicly available on Kaggle  
Released by Instacart in 2017

| File | Description |
|---|---|
| `orders.csv` | Order-level data for all customers |
| `order_products__prior.csv` | Product details for prior orders |
| `order_products__train.csv` | Product details for training orders |
| `products.csv` | Product names and IDs |
| `aisles.csv` | Aisle names |
| `departments.csv` | Department names |

📥 Download: https://www.kaggle.com/competitions/instacart-market-basket-analysis/data

> Note: The dataset files are not included in this repository due to size. Download them from Kaggle and place them in the project root directory before running.

---

## Pipeline — 7 Steps

```
orders.csv  ──┐
products.csv ─┤
aisles.csv   ─┼──► Step 1: EDA ──► Step 2: Merge ──► Step 3: Feature Engineering
departments  ─┘                                              │
                                                             ▼
                                                    Step 4: Label Engineering
                                                    (30-day churn threshold)
                                                             │
                                                             ▼
                                                    Step 5: Preprocessing
                                                    (Encode, Scale, Split)
                                                             │
                                                             ▼
                                                    Step 6: Model Training
                                                    (4 models, 5-fold CV)
                                                             │
                                                             ▼
                                                    Step 7: Risk Output
                                                    (HIGH / MEDIUM / LOW)
```

---

## Features Used

| Feature | What It Captures | Why It Matters |
|---|---|---|
| `median_days_between` | True ordering rhythm | Outlier-resistant — one long gap does not skew the picture |
| `gap_std` | Consistency of ordering | A spike in inconsistency is an early churn signal |
| `total_orders` | Customer tenure | Long-term customers churn less |
| `reorder_ratio` | Loyalty to familiar products | High ratio = habitual shopper, low ratio = exploratory |
| `avg_basket_size` | Engagement per order | Declining basket size often precedes churn |

> **Why median instead of mean?**
> A weekly shopper with gaps [7, 7, 7, 7, 7, 90] has a mean of 20.5 days — which makes them look like a biweekly shopper. Their median is 7 days — which correctly identifies them as a weekly shopper. The 90-day gap is the anomaly, not the norm. Mean punishes them for it; median does not.

---

## Churn Definition

A customer is defined as **churned** if their last recorded gap between orders is **≥ 30 days**.

**Why 30 days?**
- The dataset caps `days_since_prior_order` at 30, making anything above that unobservable
- 30 days = 4 missed weekly cycles for the most common shopper type
- Aligns with the natural "one month" business unit
- The dataset's 75th percentile gap is 15 days — 30 days is statistically unusual for virtually all customer types
- Backed by industry standard: consumable/grocery platforms use 30-day inactivity windows *(Geckoboard KPI Standards; Gorgias Ecommerce Churn Guide, 2026)*

---

## Models Trained

| Model | AUC-ROC | Notes |
|---|---|---|
| Logistic Regression | 0.717 | Baseline linear model |
| Decision Tree | 0.760 | Interpretable but tends to overfit |
| Random Forest | 0.737 | Stable ensemble |
| **XGBoost** | **0.769** | Best performance — selected as final model |

**Why AUC-ROC as the primary metric?**  
The dataset has a class imbalance (62% not churned, 38% churned). Accuracy is misleading in this case — a model that predicts "not churned" for everyone scores 62% accuracy while being completely useless. AUC-ROC evaluates ranking quality across all thresholds and is unaffected by class imbalance.

**Overfitting check:** 5-fold cross-validation on XGBoost returned Mean AUC = 0.769 ± 0.004 — the ±0.004 standard deviation confirms the model is stable and not overfitting to any particular data slice.

---

## Risk Threshold — How It Was Derived

The HIGH RISK boundary was not chosen arbitrarily. It was derived from the **Youden's J Statistic** on the ROC curve:

```
J = TPR − FPR
```

The threshold that maximises J is the statistically optimal operating point — the point where the model simultaneously catches the most churners while raising the fewest false alarms. A small precision buffer (+0.05) was added above this point to ensure that when a customer is flagged HIGH RISK, we are acting with high confidence — not just marginal probability.

---

## Methodology Notes

**Data Leakage Prevention**  
Features are computed from each customer's order history *excluding their last order*. The last order's gap is held out and used solely to generate the churn label. This mirrors real deployment: at prediction time, you only know what has already happened — not what comes next.

**Personalised Churn Window**  
Each HIGH RISK customer receives a deadline — the number of days remaining before they cross the churn threshold:

```
churn_window_days = 30 − median_days_between
```

A weekly shopper flagged today has 21 days. A customer who orders every 25 days has only 3 days. Urgency is personalised.

**Academic Basis for Churn Definition**  
The personalised approach to churn detection is grounded in:
- Fader, Hardie & Lee (2005) — *"Counting Your Customers the Easy Way"*, Marketing Science 24(2):275–284 — establishes that in non-contractual settings, churn is defined by deviation from a customer's own pattern, not a universal threshold. Free PDF: http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
- Tukey (1977) — *Exploratory Data Analysis*, Addison-Wesley — the 1.5× outlier convention that inspired the gap anomaly detection logic

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/WaliKhanJan/groceryguard
cd groceryguard
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

**3. Download the dataset**  
Download from https://www.kaggle.com/competitions/instacart-market-basket-analysis/data  
Place all `.csv` files in the project root directory.

**4. Run the notebook**
```bash
jupyter notebook Instacart_updated.ipynb
```

Run all cells top to bottom. The final cell exports `high_risk_retention_list.csv` — a ready-to-use list of every HIGH RISK customer with their recommended retention action.

---

## Output

**`high_risk_retention_list.csv`** — the retention action file

| Column | Description |
|---|---|
| `user_id` | Customer identifier |
| `top_department` | Their most-purchased department (personalise the offer here) |
| `churn_prob` | Model's predicted churn probability |
| `days_to_act` | Days remaining before this customer hits the churn threshold |
| `recommended_action` | URGENT or standard outreach, based on churn probability |

---

## Tech Stack

- **Python 3.11**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Scikit-learn** — preprocessing, model training, evaluation
- **XGBoost** — final prediction model
- **Matplotlib / Seaborn** — visualisation
- **Jupyter Notebook** — development environment

---

## Project Structure

```
groceryguard/
│
├── Instacart_updated.ipynb       # Main notebook — full pipeline
├── high_risk_retention_list.csv  # Output: actionable retention list (generated on run)
├── README.md                     # This file
│
└── data/                         # Place Kaggle CSV files here
    ├── orders.csv
    ├── order_products__prior.csv
    ├── order_products__train.csv
    ├── products.csv
    ├── aisles.csv
    └── departments.csv
```

---

## Author

**Wali Muhammad Nasir**  
AI/ML Engineer | Data Intern @ Convergent Business Technologies  
[LinkedIn](https://www.linkedin.com/in/wali-muhammad-nasir) · [GitHub](https://github.com/WaliKhanJan)

---

## References

1. Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005). "Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model." *Marketing Science*, 24(2), 275–284. http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
2. Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley. https://archive.org/details/exploratorydataa0000tuke_7616
3. Decile (2023). *Ecommerce Benchmarking Guide*. https://decile.com
4. Instacart (2017). *The Instacart Online Grocery Shopping Dataset 2017*. https://www.kaggle.com/competitions/instacart-market-basket-analysis

