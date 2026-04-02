# Glass-Box Churn Prediction (XAI-First)

A practical, notebook-first study on customer churn prediction that directly compares **post-hoc explainability** against **intrinsic (glass-box) explanations**. The core workflow lives in `GlassBox.ipynb` and evaluates three model families—LightGBM + SHAP, EBM, and TabTransformer—across **accuracy, latency, fidelity, and stability**.

## Project Title
**A Glass-Box Pipeline for Customer Churn Prediction: Comparing Intrinsic EBM Explanations Against Post-Hoc SHAP and Attention-Based TabTransformer Across Accuracy, Latency, Fidelity, and Stability**

## Research Gap Addressed
The base paper, **“A Big Data-Driven Hybrid Model for Enhancing Streaming Service Customer Retention Through Churn Prediction Integrated With Explainable AI”**, relies on post-hoc XAI (e.g., SHAP/LIME). This project targets three key gaps:

1. **Post-Hoc Latency Gap (Computational Efficiency)**  
   Post-hoc methods can be too slow for real-time analytics. We benchmark explanation time and show why intrinsic methods are better suited for high-velocity business pipelines.
2. **Fidelity–Explanation Gap**  
   Post-hoc explainers approximate black-box behavior and can be misleading. We measure fidelity and contrast it with intrinsically faithful explanations.
3. **Static vs. Dynamic Explanations (Concept Drift)**  
   Explanations can become stale as customer behavior evolves. The notebook frames why models that explain themselves are more robust under drift.

**Proposed Research Angle:**  
Instead of “black-box + post-hoc”, this work argues for **glass-box pipelines** such as EBM or attention-based tabular transformers. The aim is **high accuracy with real-time, high-fidelity explanations**.

## What’s In This Repo
- `GlassBox.ipynb`  
  End-to-end pipeline: data acquisition, preprocessing, EDA, feature engineering, SMOTE, model training, explainability, metrics, and final findings.
- EDA visuals  
  `eda_plot_1.png` … `eda_plot_6.png`
- Model artifacts and diagnostics  
  `lgbm_model.pkl`, `lgbm_learning_curve.png`
- Comparison figures  
  `metric1_performance.png`, `metric2_latency.png`, `metric3_fidelity.png`, `metric4_stability*.png`, `smote_class_distribution.png`

## How To Run
This is a notebook-driven project.

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run `GlassBox.ipynb` top-to-bottom.

### Suggested Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install numpy pandas scikit-learn lightgbm shap interpret imbalanced-learn torch pytorch-tabular matplotlib seaborn scipy statsmodels tqdm ucimlrepo pillow joblib
```

### Run the Notebook
```bash
jupyter notebook
```
Open `GlassBox.ipynb` and run all cells.

## Notebook Modules (High-Level)
1. Environment setup  
2. Dataset acquisition (UCI Online Retail II)  
3. Preprocessing & churn label engineering  
4. EDA  
5. Feature engineering & pipeline construction  
6. SMOTE for class imbalance  
7. Baseline: LightGBM + SHAP  
8. Proposed: Explainable Boosting Machine (EBM)  
9. Advanced: TabTransformer (attention-based)  
10. Metrics: Accuracy, Latency, Fidelity, Stability  
11. Master comparison & statistical testing  
12. Final figures & per-customer explanation deep dive  
13. Research findings summary

## Key Findings (From `MODULE 15`)
1. **Predictive performance**  
   All three models reached near-perfect performance on the synthetic dataset. McNemar’s test showed **no significant performance difference** between LightGBM and EBM for this setup.
2. **Explanation latency**  
   EBM explanations were **~1.9x faster** than LightGBM + SHAP on the full test set.
3. **Explanation fidelity**  
   EBM is **perfectly faithful** by design (fidelity = 1.0).  
   SHAP’s composite fidelity was **0.7618**, reflecting approximation errors.
4. **Explanation stability**  
   Both EBM and TreeSHAP were **deterministic**, resulting in perfect stability under identical inputs.
5. **TabTransformer**  
   Strong predictive performance (AUC-ROC 0.992, accuracy 0.958). Attention maps offer a distinct interpretability lens for feature interactions.

**Bottom line:** EBM offers the best balance of **accuracy + speed + fidelity + stability** for churn prediction in a glass-box pipeline, while TabTransformer remains promising for attention-based insight.

## Literature Foundation
### 10 Core Papers 
1. Predicting E-commerce Customer Churn with XAI (IEEE Access/Pre-print 2025)  
2. HPrEd: A Tuned Ensemble With Model-Agnostic XAI to Explain Social Media’s Association With Academic Productivity (IEEE Xplore 2026)  
3. Explainable AI-Driven Intrusion Detection System (IEEE Xplore 2026)  
4. A Literature Review on Applications of Explainable Artificial Intelligence (XAI) (IEEE Access 2025)  
5. Predictive Analytics in Human Resources Management: Evaluating AIHR's Role in Talent Retention (IEEE/MDPI 2025)  
6. Food Demand Prediction Using Nonlinear Autoregressive Exogenous Neural Networks (IEEE Xplore 2024)  
7. A Survey of Industrial AIoT: Opportunities and Directions (IEEE Xplore 2025)  
8. Analyzing False Positives with Explainable AI (IEEE Xplore 2023)  
9. Machine Learning for Customer Churn Prediction in Retail (IEEE-indexed/RG 2024)  
10. Navigating the Trade-offs: Accuracy vs. Interpretability in ML Models (IEEE-related/AIMS 2025)

### 20 Supporting/Smaller Papers (Provided Reference List)
1. XGBoost vs. LightGBM for E-commerce Prediction (2024)  
2. Local Explanations for User Retention using LIME (2023)  
3. Feature Importance in High-Dimensional Retail Data (2022)  
4. Deep Learning for Sales Forecasting in Fragmented Markets (2025)  
5. Visualizing Black-Box Decisions in Business Intelligence (2024)  
6. The Role of Counterfactual Explanations in Customer Behavior (2026)  
7. Impact of Data Quality on XAI Reliability (2024)  
8. Optimizing Hyperparameters for Explainable XGBoost (2023)  
9. SMOTE Techniques for Imbalanced Churn Datasets (2025)  
10. Evaluating Trustworthiness of AI in E-retail (2024)  
11. A Comparative Study of Kernel SHAP vs. Tree SHAP (2023)  
12. Interpretable Neural Networks for Demand Forecasting (2022)  
13. XAI for Real-time Inventory Management (2025)  
14. Measuring User Trust in AI-generated Explanations (2024)  
15. Algorithmic Fairness in Retail Recommendation Systems (2026)  
16. SHAP-based Analysis of Holiday Sales Peaks (2023)  
17. Reducing Computational Cost of LIME in Large Datasets (2024)  
18. Case Study: XAI for Subscription-based Business Models (2025)  
19. Comparing Permutation Importance vs. SHAP (2022)  
20. Human-in-the-loop: Validating XAI Results with Domain Experts (2026)

## Suggested Next Steps
1. Add a `requirements.txt` or `pyproject.toml` for reproducibility.  
2. Expand the TabTransformer attention analysis to quantify feature interaction stability over time.  
3. Evaluate drift-aware explainers using rolling windows or temporal splits.
