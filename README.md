# How sharp is Shap?
## Explainable ML for Job-Skill Matching, Sentiment & Click Prediction
## Introduction

When we heard the question “Does your therapist need a therapist?” – we immediately thought: well, maybe SHAP does.  
In this project, we ask if explainable AI methods like **SHAP** and **LIME** truly explain tree-based models, or just throw around plausible-sounding scores.  
To find out, we ran both methods across three datasets —  
- **Click Prediction** (predicting ad engagement)  
- **Job–Skill Matching** (from Glassdoor listings)  
- **Sentiment Analysis** (based on job description text)  


and three models: 
- **Random Forest** 
- **XGBoost**
- **BERT**
We then evaluated them using five custom metrics — because sometimes, explainability needs explaining.


---

## Table of Contents

- [Introduction](#introduction)  
- [Research Questions](#research-questions)  
- [SHAP vs. LIME: Key Differences](#shap-vs-lime-key-differences-)  
- [Evaluation Framework](#4-evaluation-framework)  
- [Datasets](#datasets)  
- [5. Experimental Setup](#5-experimental-setup)  
  - [Models](#models)  
  - [Datasets and Tasks](#datasets-and-tasks)  
- [Explainability](#explainability)  
  - [Sub-RQ1: Global Explanation Quality](#sub-rq1-global-explanation-quality)  
  - [Sub-RQ2: Local Explanation Consistency and Agreement](#sub-rq2-local-explanation-consistency-and-agreement)  
  - [Sub-RQ3: Monotonicity of Key Features](#sub-rq3-monotonicity-of-key-features)  
  - [Sub-RQ4: Cross-Model and Cross-Domain Comparison](#sub-rq4-cross-model-and-cross-domain-comparison)  
- [Final Summary](#final-summary)  

---

## Research Questions
To figure out whether SHAP and LIME are actually doing their job (or just sounding smart), we broke down our investigation into one main question and four sub-questions. Each one targets a specific aspect of explainability — from global insights to local stability — across different models and domains. Here's how we framed it:

**Main RQ**  
> How effectively can SHAP and LIME explain complex tree-based models across different domains (job–skill matching, sentiment analysis, click prediction)?

**Sub-RQ 1: Global Explanation Quality**  
> Which features do SHAP’s global importance scores identify as most predictive, and how faithfully do those global rankings reflect actual model behavior?  
> - **Metrics:**  
>   - **Global Fidelity (MEMC):** average AUC drop when zeroing out top-k SHAP features  
>   - **Top-k AUPRC:** precision of a fresh classifier trained only on the top-k SHAP features  

**Sub-RQ 2: Local Explanation Consistency & Agreement**  
> How stable and consistent are local explanations under perturbations, and how well do SHAP and LIME agree on individual predictions?  
> - **Stability:** Infidelity & Sensitivity under small input perturbations  
> - **Agreement:** Spearman’s ρ between SHAP vectors and LIME weights  

**Sub-RQ 3: Monotonicity of Key Features**  
> Do SHAP attributions vary monotonically with crucial continuous inputs (e.g. average salary, sentiment score, click probability)?  
> - **Metric:** Spearman’s ρ between feature-quantile bins and mean SHAP in each bin  

**Sub-RQ 4: Cross-Model & Cross-Domain Comparison**  
> How do explanation-quality metrics compare:  
> 1. Random Forest vs. XGBoost  
> 2. Job–Skill vs. Sentiment vs. click datasets  

---
## SHAP vs. LIME: Key Differences ✅❌⚠️

| Feature                     | SHAP                                                                 | LIME                                                                 |
|----------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| **Model Access**           | Model-specific (needs internals) ✅                                  | Model-agnostic (black-box) ✅                                        |
| **Method**                 | Shapley values from game theory ✅                                   | Local linear approximation ⚠️                                       |
| **Explanation Scope**      | Global + local ✅                                                    | Local only ❌                                                       |
| **Interpretability**       | More accurate but complex ⚠️                                         | Very intuitive ✅                                                    |
| **Stability**              | Generally stable ✅                                                  | Unstable with small changes ❌                                       |
| **Computation Time**       | Slower, especially for large datasets ❌                             | Faster and lightweight ✅                                            |
| **Faithfulness to Model**  | Very high (captures interactions) ✅                                 | Approximate; may miss key logic ⚠️                                  |
| **Best Use Case**          | When precision matters, esp. with tree models ✅                     | Quick insights when model internals aren’t available ✅              |

## 4. Evaluation Framework

To figure out if SHAP and LIME actually explain what the models are doing (and not just pretend to), we used five core metrics — each targeting a different aspect of explanation quality.

| **Metric** | **What It Measures** | **Why It Matters** | **In Plain English** |
|------------|-----------------------|---------------------|------------------------|
| **MEMC** (Mean Excluded Marginal Contribution) | How much model performance drops when top SHAP features are removed. | Tells us whether SHAP is pointing to the features that *actually* drive the model. | “If we hide the features SHAP says are important, does the model get worse?” |
| **AUPRC (Top-k Feature Selection)** | How well a model performs using only the top-k SHAP-ranked features. | Checks if SHAP-selected features are strong enough to stand alone. | “Can you build a good model using *only* SHAP’s top picks?” |
| **SHAP–LIME Agreement** | Rank correlation between SHAP and LIME attributions for the same prediction (Spearman’s ρ). | Evaluates consistency between explanation methods. | “Do SHAP and LIME agree on what mattered in one prediction?” |
| **Robustness** (Infidelity & Sensitivity) | Infidelity: how well SHAP matches model behavior under small changes.<br>Sensitivity: how much SHAP values change when input is slightly perturbed. | Tells us how stable the explanation is. | “If we shake the input a bit, does the explanation still hold up?” |
| **Monotonicity** | Whether SHAP values increase/decrease consistently with a continuous input (e.g. salary). | Helps check if explanations behave intuitively. | “If salary goes up, does its SHAP value go up too — as expected?” |

## Datasets

## 5. Experimental Setup

We trained two popular tree-based models — **Random Forest** and **XGBoost** — across three datasets, each representing a distinct domain.  
The goal: apply SHAP and LIME, then evaluate how “explainable” their explanations really are.

- **Random Forest** served as our baseline: it’s widely used, interpretable, and relatively stable — a good benchmark for comparing explanation quality.
- **XGBoost**, on the other hand, is powerful but often seen as a black box — especially in high-dimensional data. We wanted to know: can SHAP keep up?

All models were trained on consistent train/test splits (80/20), with careful preprocessing and feature engineering applied beforehand.

### Models

| Model            | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Random Forest**| Interpretable, robust, spreads feature importance evenly — used as baseline.|
| **XGBoost**      | High-performance, non-linear and harder to interpret — a good test for SHAP/LIME.|

### Datasets and Tasks

| **Dataset**             | **Prediction Task**                             | **Target**               | **Data Type**               | **Source**                                                                                  |
|-------------------------|--------------------------------------------------|---------------------------|------------------------------|---------------------------------------------------------------------------------------------|
| **Job–Skill Matching**  | Predict experience level from job descriptions   | `avg_salary` (regression)| Structured + Text            | [Glassdoor (2017–2018)](https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor) |
| **Sentiment Analysis**  | Classify job ads as positive or negative         | `sentiment_label` (binary)| Text (n-grams or embeddings) | [Glassdoor (2017–2018)](https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor) |
| **Click Prediction**    | Predict if a user clicks on a job ad             | `clicked` (binary)        | Categorical + Numeric        | [criteo/FairJob](https://huggingface.co/datasets/criteo/FairJob)                            |

Each model–dataset combination was explained using both SHAP and LIME, then scored along five custom metrics to evaluate global and local explanation quality.


---
# Explainability 

## Sub-RQ1: Global Explanation Quality
Which features do SHAP’s global importance scores identify as most predictive, and how faithfully do those global rankings reflect actual model behavior?

### MEMC (Mean Excluded Marginal Contribution)  
SHAP’s global feature rankings were most faithful in the FairJob dataset, where removing top-ranked features led to clear drops in model performance — especially in XGBoost, with MEMC of 0.137 vs. 0.114 for RF. ✅

In the Skill → Salary task, high MEMC values for features like Python (0.247) and AWS (0.286) in XGBoost showed strong model reliance on these features. However, RF models showed lower MEMC, indicating more diffuse reliance. ⚠️

In the Job Description → Experience task, MEMC scores were relatively low for both models (0.18 and ~0.25), suggesting weaker alignment between SHAP scores and actual model behavior. ⚠️❌

### AUPRC (Top-k Feature Selection)  
Across datasets, AUPRC confirmed that SHAP’s top features preserved predictive power:

In FairJob, AUPRC@3 reached 0.803 (XGBoost) and 0.752 (RF), reflecting high global fidelity. ✅

In the Skill task, AUPRC@3 exceeded 0.84 for Python and Excel, again confirming that top SHAP features carried strong signals. ✅

However, in the Job Description dataset, AUPRC for the top feature (salary) was only 0.64, with lower MEMC — suggesting only partial faithfulness. ⚠️

Conclusion: 

SHAP’s global importance scores were generally predictive and informative, especially in structured tasks like click prediction and skill impact.  
Yet, their faithfulness varied across datasets and models — strong in **XGBoost** and **FairJob**, but weaker in **Random Forest** and **text-heavy tasks**.  
**Overall rating:** ✅✅⚠️

---

## Sub-RQ2: Local Explanation Consistency and Agreement
How stable and consistent are local explanations under perturbations, and how well do SHAP and LIME agree on individual predictions?
### SHAP vs. LIME (Spearman Correlation)  
Local agreement between SHAP and LIME was consistently weak or negative across all datasets:

In the FairJob dataset, agreement was particularly poor, with Spearman ρ ≈ –0.72 (RF) and –0.45 (XGBoost), reflecting fundamentally different local explanations. ❌

In the Skill → Salary task, agreement scores ranged from 0.16 to 0.22, showing some convergence but not strong alignment. ⚠️

This suggests that SHAP and LIME rarely point to the same top features on an instance level — a critical gap for explainability in user-facing applications. ❌

### Infidelity  
SHAP explanations were generally faithful to the model’s output, especially in Random Forest models:

Across all datasets, RF models had lower infidelity scores than XGBoost (e.g., 0.0025 vs. 0.0056 in FairJob), showing that SHAP matched model behavior more closely. ✅

In Skill → Salary, XGBoost infidelity rose significantly (e.g., 0.103 for Python), suggesting unstable explanations despite high feature importance. ⚠️❌
### Sensitivity  
Stability under small input perturbations was highly model-dependent:

RF models were far more stable, with sensitivity values close to zero — e.g., 0.0015 in FairJob — indicating robust local explanations. ✅

XGBoost models, however, were highly sensitive, especially in the Skill task (e.g., 57.5 for Spark, 49.6 for Python), meaning small changes to input led to large changes in explanation. ❌  
**Conclusion:**  
Local SHAP explanations were stable and faithful in **Random Forest** models, but sensitive and inconsistent in **XGBoost**.  
**SHAP–LIME agreement** was consistently poor, undermining trust in explanations across methods.  
**Overall rating:** ✅⚠️❌

---

## Sub-RQ3: Monotonicity of Key Features
Monotonicity of SHAP values with respect to key continuous features varied significantly across datasets and models:

In the Skill → Salary task, SHAP values for features like Python (0.95) and AWS (0.92) in Random Forest and XGBoost showed strong monotonic relationships — i.e., as the feature value increased, its SHAP contribution also consistently increased. ✅

In the FairJob dataset, monotonicity was moderate (e.g., ~0.49 for XGBoost, ~0.45 for RF), indicating partial alignment between input strength (e.g., salary, click probability) and model explanation. ⚠️

In the Job Description → Experience task, monotonicity scores were very low (~0.15–0.20), suggesting little to no consistent pattern between input changes and SHAP attributions. ❌

Conclusion

SHAP attributions were strongly monotonic in well-structured, numeric tasks like skill-based salary prediction, but less consistent in more abstract or text-based settings.
Overall rating: ✅⚠️❌

---

## Sub-RQ4: Cross-Model and Cross-Domain Comparison
| Dataset                     | Target       | Model    | SHAP MEMC | AUPRC@3 | Infidelity | Sensitivity | Monotonicity | SHAP–LIME Agreement |
|-----------------------------|--------------|----------|-----------|---------|-------------|--------------|---------------|----------------------|
| Salary Data (Job Description) | Avg Salary   | RF       | ⚠️ 0.18   | ⚠️ 0.64 | ✅ 0.0067   | ⚠️ 0.78     | ❌ 0.15        | ❌ Low / unstable     |
|                             |              | XGBoost  | ✅ ~0.25   | ✅ 0.71 | ❌ 0.0090   | ❌ 0.82     | ❌ 0.20        | ❌ Low                |
| FairJob                     | Click        | RF       | ⚠️ 0.114  | ✅ 0.752| ✅ 0.0025   | ✅ 0.0015   | ⚠️ 0.455       | ❌ -0.722             |
|                             |              | XGBoost  | ✅ 0.137   | ✅ 0.803| ⚠️ 0.0056   | ❌ 0.0108   | ⚠️ 0.492       | ❌ -0.445             |
| Salary Data (Skill → Salary)| Python       | RF       | ⚠️ 0.151  | ✅ 0.846| ✅ 0.0006   | —           | ✅ 0.95        | ⚠️ 0.18               |
|                             |              | XGBoost  | ✅ 0.247   | ✅ 0.887| ❌ 0.103    | ❌ 49.64    | ⚠️ 0.39        | ⚠️ 0.18               |
|                             | Spark        | RF       | ❌ 0.017   | ✅ 0.781| ✅ 0.0005   | —           | ✅ 0.81        | ⚠️ 0.22               |
|                             |              | XGBoost  | ⚠️ 0.068   | ⚠️ 0.618| ❌ 0.076    | ❌ 57.51    | ❌ -0.09       | ⚠️ 0.22               |
|                             | AWS          | RF       | ⚠️ 0.046   | ✅ 0.798| ✅ 0.0008   | —           | ✅ 0.82        | ⚠️ 0.16               |
|                             |              | XGBoost  | ✅ 0.286   | ⚠️ 0.700| ❌ 0.062    | ❌ 42.46    | ✅ 0.92        | ⚠️ 0.16               |
|                             | Excel        | RF       | ⚠️ 0.089   | ✅ 0.799| ✅ 0.0008   | —           | ❌ -0.78       | ⚠️ 0.19               |
|                             |              | XGBoost  | ⚠️ 0.162   | ✅ 0.759| ❌ 0.122    | ❌ 57.53    | ❌ -0.59       | ⚠️ 0.19               |


These comparisons show that SHAP provides more detailed and accurate explanations than LIME, especially in XGBoost models, where top-ranked features aligned well with model behavior. However, this came at the cost of greater sensitivity and reduced local stability. Random Forest models offered more robust and consistent explanations but tended to distribute importance more evenly, leading to lower global faithfulness.
---

## Final Summary
SHAP emerged as a more effective and reliable explainability technique than LIME across nearly all evaluation criteria.
It produced high-fidelity global feature rankings and locally faithful attributions, even in complex, high-dimensional tasks. SHAP’s alignment with model behavior was particularly strong in XGBoost, which also achieved the highest predictive performance and most concentrated feature reliance.

However, SHAP explanations were also more sensitive in XGBoost, reflecting its sharper decision boundaries and susceptibility to input perturbations.
By contrast, Random Forest offered more robust and stable explanations, albeit with less feature concentration.

LIME, on the other hand, consistently showed weak to negative agreement with SHAP and less reliable local explanations — especially in low-dimensional or text-based contexts.

In conclusion, SHAP is the preferred tool for explaining tree-based models across diverse domains. It enables trustworthy explanations that support both global interpretation and instance-level reasoning — making it a strong choice for practical applications like job recommendations, skill-based salary modeling, and click prediction. ✅

In short, if you’re trusting a model to make decisions, SHAP is what you want explaining those decisions.
---


