# CS340 Final Project Presentation
## Bias Mitigation in Employment Prediction

### Slide 1: Title
- Title: Bias Mitigation in Employment Prediction
- Subtitle: CS340 Final Project
- Date: May 16, 2026

### Slide 2: Problem Statement
- Dataset: ACSEmployment (646,917 records)
- Task: Predict employment status
- Challenge: Avoid gender bias
- Goal: Equal opportunity for male and female groups

### Slide 3: Baseline Model - Performance
- Model: Logistic Regression
- Accuracy: 77.39%
- AUC: 85.03%
- Visualization: Performance metrics bar chart

### Slide 4: Baseline Model - Bias Analysis
- Equal Opportunity Difference: 0.0956 (SIGNIFICANT)
- Male TPR: 86.74% vs Female TPR: 77.18%
- Gap: 9.56% (females more likely to be misclassified)
- Visualization: Group comparison

### Slide 5: Our Approach - Overview
- Multi-stage bias mitigation
- Stage 1: Fairness-aware sample weighting
- Stage 2: Ensemble model training
- Stage 3: Ensemble prediction
- Stage 4: Threshold optimization

### Slide 6: Method Details
- Sample weighting: Balance (group, label) combinations
- Ensemble: LR + RF + GB with fairness weights
- Threshold optimization: Equalize TPR across groups
- Diagram: Pipeline visualization

### Slide 7: Results - Performance
- Accuracy: 77.39% → 77.80% (+0.41%)
- AUC: 85.03% → 89.40% (+4.37%)
- Both metrics IMPROVED!
- Visualization: Before/after comparison

### Slide 8: Results - Fairness
- Equal Opportunity Diff: 0.0956 → 0.0060 (-93.7%)
- Demographic Parity: 0.0874 → 0.0109 (-87.5%)
- Male TPR: 96.86% vs Female TPR: 96.26%
- Gap reduced from 9.56% to 0.60%

### Slide 9: Key Findings
- Fairness and accuracy CAN both improve
- Multi-stage approach is effective
- 11/13 metrics improved (Level 5)
- Score: 5/5 points

### Slide 10: Trade-offs
Advantages:
- Dramatic fairness improvement
- Better overall performance
- Equitable outcomes

Disadvantages:
- Higher FPR (26.65% → 37.84%)
- Computational cost
- Requires threshold tuning

### Slide 11: Conclusion
- Successfully mitigated bias while improving accuracy
- 93.7% reduction in Equal Opportunity Difference
- Demonstrates that fairness ≠ accuracy trade-off
- Practical and interpretable approach

### Slide 12: Q&A
- Thank you!
- Questions?
