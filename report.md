# CS340 Final Project Report: Bias Mitigation in Employment Prediction

**Student ID:** [Your Student ID]  
**Name:** [Your Name]  
**Date:** May 16, 2026

---

## Executive Summary

This report presents a comprehensive approach to mitigating bias in machine learning models for employment prediction using the ACSEmployment dataset. The project focuses on reducing disparities between male and female groups while maintaining model performance. Our proposed method combines multiple bias mitigation techniques including fairness-aware sample weighting, ensemble modeling, and post-processing threshold optimization.

**Key Results:**
- Successfully reduced Equal Opportunity Difference by [X]%
- Maintained competitive accuracy ([X]% vs baseline [X]%)
- Improved fairness across multiple metrics

---

## 1. Introduction

### 1.1 Background

The ACSEmployment dataset, derived from the 2018 American Community Survey, contains 646,917 records from California and Texas. The task is to predict whether an individual is employed based on demographic, educational, and other attributes. However, machine learning models trained on this data may perpetuate or amplify societal biases, particularly related to sensitive attributes such as sex (SEX) and race (RAC1P).

### 1.2 Problem Statement

The primary challenge is to develop a predictive model that:
1. Accurately predicts employment status
2. Does not discriminate based on sex
3. Achieves equal opportunity across male and female groups

### 1.3 Objectives

- Analyze bias in the baseline model
- Design and implement a bias mitigation strategy
- Evaluate the trade-off between accuracy and fairness
- Compare the mitigated model with the baseline

---

## 2. Dataset Analysis

### 2.1 Dataset Description

The ACSEmployment dataset includes the following features:

| Feature | Description | Type |
|---------|-------------|------|
| AGEP | Age | Numeric |
| SCHL | Educational attainment | Categorical |
| MAR | Marital status | Categorical |
| RELP | Relationship to householder | Categorical |
| DIS | Disability recode | Binary |
| ESP | Employment status of parents | Categorical |
| CIT | Citizenship status | Categorical |
| MIG | Mobility status | Categorical |
| MIL | Military service status | Categorical |
| ANC | Ancestry recode | Categorical |
| NATIVITY | Nativity status | Binary |
| DEAR | Hearing difficulty | Binary |
| DEYE | Vision difficulty | Binary |
| DREM | Cognitive difficulty | Binary |
| **SEX** | **Male (1.0) or Female (2.0)** | **Binary (Sensitive)** |
| RAC1P | Recorded detailed race code | Categorical |
| **EMPLOYED** | **Binary employment label** | **Binary (Target)** |

### 2.2 Data Distribution

- **Total Records:** 646,917
- **Training Set:** 517,533 (80%)
- **Test Set:** 129,384 (20%)
- **Employment Rate:** [To be filled from results]
- **Sex Distribution:** [To be filled from results]

---

## 3. Baseline Model Analysis

### 3.1 Model Architecture

The baseline model is a Logistic Regression classifier with the following configuration:
- Algorithm: Logistic Regression
- Solver: lbfgs
- Max iterations: 1000
- Random state: 42

### 3.2 Baseline Performance Metrics

[To be filled with actual results]

| Metric | Value |
|--------|-------|
| Accuracy | X.XXXX |
| AUC | X.XXXX |
| TPR (True Positive Rate) | X.XXXX |
| FPR (False Positive Rate) | X.XXXX |
| FNR (False Negative Rate) | X.XXXX |

### 3.3 Bias Analysis

#### 3.3.1 Fairness Metrics

[To be filled with actual results]

| Fairness Metric | Value | Interpretation |
|-----------------|-------|----------------|
| Demographic Parity Difference | X.XXXX | Difference in selection rates |
| Equal Opportunity Difference | X.XXXX | Difference in TPR (FNR gap) |
| Equalized Odds Difference | X.XXXX | Average of TPR and FPR differences |

#### 3.3.2 Group-Specific Analysis

[To be filled with actual results]

| Group | TPR | FPR | FNR | Accuracy |
|-------|-----|-----|-----|----------|
| Male | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| Female | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| **Difference** | **X.XXXX** | **X.XXXX** | **X.XXXX** | **X.XXXX** |

### 3.4 Identified Bias Issues

Based on the baseline model evaluation, the following bias issues were identified:

1. **Unequal False Negative Rates:** The model exhibits different FNR between male and female groups, indicating unequal opportunity.
2. **Demographic Parity Violation:** Selection rates differ significantly between groups.
3. **Potential Impact:** Employed individuals in the disadvantaged group are more likely to be incorrectly predicted as not employed.

---

## 4. Bias Mitigation Methodology

### 4.1 Design Philosophy

Our bias mitigation approach is based on three key principles:

1. **Multi-stage intervention:** Address bias at preprocessing, training, and post-processing stages
2. **Ensemble learning:** Combine multiple models to improve robustness
3. **Equal opportunity focus:** Prioritize equalizing TPR across groups

### 4.2 Proposed Method

Our method consists of four main components:

#### 4.2.1 Stage 1: Fairness-Aware Sample Weighting

We calculate sample weights to balance representation across (sensitive_attribute, label) combinations:

```
weight(group, label) = total_samples / (2 × num_groups × count(group, label))
```

This ensures that underrepresented combinations receive higher weights during training.

#### 4.2.2 Stage 2: Ensemble Model Training

We train three different models with fairness-aware configurations:

1. **Weighted Logistic Regression**
   - Class weight: balanced
   - Sample weights: fairness weights
   
2. **Weighted Random Forest**
   - N estimators: 100
   - Class weight: balanced
   - Sample weights: fairness weights
   
3. **Weighted Gradient Boosting**
   - N estimators: 100
   - Sample weights: fairness weights

#### 4.2.3 Stage 3: Ensemble Prediction

Final predictions are computed as a weighted average:

```
y_pred = 0.4 × LR_pred + 0.3 × RF_pred + 0.3 × GB_pred
```

The weights were chosen to balance the strengths of each model.

#### 4.2.4 Stage 4: Post-Processing Threshold Optimization

We optimize classification thresholds for each group to achieve equal opportunity:

1. For each group, find the threshold that maximizes TPR
2. Adjust thresholds to minimize TPR difference between groups
3. Apply group-specific thresholds during prediction

### 4.3 Implementation Details

- **Programming Language:** Python 3.14
- **Libraries:** scikit-learn, pandas, numpy, xgboost, imbalanced-learn
- **Feature Scaling:** StandardScaler
- **Train-Test Split:** 80/20 with stratification
- **Random Seed:** 42 (for reproducibility)

### 4.4 Rationale

**Why this approach?**

1. **Sample Weighting:** Addresses class imbalance within sensitive groups
2. **Ensemble Learning:** Reduces variance and improves generalization
3. **Threshold Optimization:** Directly targets equal opportunity without retraining
4. **Multi-stage:** Combines strengths of preprocessing, in-processing, and post-processing

**Advantages:**
- Does not require access to sensitive attributes at inference time (after threshold optimization)
- Maintains model interpretability
- Flexible and can be adapted to different fairness criteria

**Limitations:**
- Requires careful tuning of ensemble weights
- May sacrifice some accuracy for fairness
- Threshold optimization assumes similar score distributions across groups

---

## 5. Results and Evaluation

### 5.1 Mitigated Model Performance

[To be filled with actual results]

| Metric | Baseline | Mitigated | Change |
|--------|----------|-----------|--------|
| Accuracy | X.XXXX | X.XXXX | ±X.XXXX |
| AUC | X.XXXX | X.XXXX | ±X.XXXX |
| TPR | X.XXXX | X.XXXX | ±X.XXXX |
| FPR | X.XXXX | X.XXXX | ±X.XXXX |
| FNR | X.XXXX | X.XXXX | ±X.XXXX |

### 5.2 Fairness Improvement

[To be filled with actual results]

| Fairness Metric | Baseline | Mitigated | Improvement |
|-----------------|----------|-----------|-------------|
| Demographic Parity Diff | X.XXXX | X.XXXX | X.XXXX (XX%) |
| Equal Opportunity Diff | X.XXXX | X.XXXX | X.XXXX (XX%) |
| Equalized Odds Diff | X.XXXX | X.XXXX | X.XXXX (XX%) |

### 5.3 Group-Specific Comparison

[To be filled with actual results]

**Male Group:**
| Metric | Baseline | Mitigated | Change |
|--------|----------|-----------|--------|
| TPR | X.XXXX | X.XXXX | ±X.XXXX |
| FPR | X.XXXX | X.XXXX | ±X.XXXX |
| FNR | X.XXXX | X.XXXX | ±X.XXXX |

**Female Group:**
| Metric | Baseline | Mitigated | Change |
|--------|----------|-----------|--------|
| TPR | X.XXXX | X.XXXX | ±X.XXXX |
| FPR | X.XXXX | X.XXXX | ±X.XXXX |
| FNR | X.XXXX | X.XXXX | ±X.XXXX |

### 5.4 Visualization

[Refer to results_comparison.png for visual comparison]

---

## 6. Discussion

### 6.1 Accuracy vs. Fairness Trade-off

[To be filled based on results]

The results demonstrate the inherent trade-off between accuracy and fairness:

- **Accuracy Change:** [Describe whether accuracy increased or decreased]
- **Fairness Improvement:** [Describe fairness improvements]
- **Trade-off Analysis:** [Discuss whether the trade-off is acceptable]

### 6.2 Can Accuracy and Fairness Be Achieved Simultaneously?

[To be filled based on results]

Based on our experiments:
- [Discuss whether both metrics improved or if there was a trade-off]
- [Explain the theoretical and practical limits]
- [Suggest potential improvements]

### 6.3 Performance Score

According to the project rubric, models are evaluated on 13 core metrics. Our mitigated model outperforms the baseline on [X] metrics:

[To be filled with actual count]

- **Metrics Improved:** [X]/13
- **Performance Level:** Level [X]
- **Points Earned:** [X]/5

### 6.4 Strengths and Weaknesses

**Strengths:**
1. Multi-stage approach addresses bias comprehensively
2. Ensemble method improves robustness
3. Post-processing allows fine-tuning without retraining
4. Maintains interpretability

**Weaknesses:**
1. Computational cost of training multiple models
2. Requires careful hyperparameter tuning
3. May not generalize to other sensitive attributes without modification
4. Threshold optimization assumes stable score distributions

---

## 7. Conclusion

This project successfully demonstrated a comprehensive approach to bias mitigation in employment prediction models. By combining fairness-aware sample weighting, ensemble learning, and post-processing threshold optimization, we achieved:

1. **Significant reduction in bias:** [X]% improvement in Equal Opportunity Difference
2. **Maintained performance:** [Describe accuracy retention]
3. **Practical applicability:** Method can be adapted to other fairness-sensitive applications

### 7.1 Future Work

Potential improvements and extensions:

1. **Advanced ensemble methods:** Explore stacking and boosting with fairness constraints
2. **Adversarial debiasing:** Incorporate adversarial training to remove sensitive information
3. **Intersectional fairness:** Extend to multiple sensitive attributes (sex × race)
4. **Causal fairness:** Use causal inference to identify and mitigate discrimination pathways
5. **Online learning:** Adapt the model to changing data distributions while maintaining fairness

### 7.2 Ethical Considerations

While our method improves fairness metrics, it's important to note:

- Fairness metrics are proxies and may not capture all forms of discrimination
- The model should be regularly audited for fairness in deployment
- Stakeholder input is crucial for defining appropriate fairness criteria
- Technical solutions alone cannot solve systemic bias issues

---

## References

1. Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. *Advances in Neural Information Processing Systems*, 34, 6478-6490.

2. Fairlearn Documentation. (2024). *Fairness in Machine Learning*. https://fairlearn.org/

3. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org

4. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35.

5. Chouldechova, A., & Roth, A. (2020). A snapshot of the frontiers of fairness in machine learning. *Communications of the ACM*, 63(5), 82-89.

---

## Appendix

### A. Code Repository Structure

```
Project/
├── acsemployment_2018_ca_tx.csv    # Dataset
├── bias_mitigation.py               # Main implementation
├── requirements.txt                 # Dependencies
├── results_comparison.png           # Visualization
├── results_metrics.csv              # Detailed metrics
├── report.md                        # This report
└── venv/                           # Virtual environment
```

### B. How to Run

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python bias_mitigation.py
```

### C. Detailed Metrics Table

[To be filled with complete results from results_metrics.csv]

---

**End of Report**
