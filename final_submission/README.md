# Fair Employment Prediction Model

This project implements a fair machine learning model for employment prediction that mitigates gender bias while maintaining high accuracy.

## Project Structure

```
final_submission/
├── baseline_model.py          # Baseline neural network (no fairness constraints)
├── fair_model.py              # Fair model with adversarial debiasing
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── report.pdf                 # Detailed project report
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- See `requirements.txt` for complete dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Download the ACS Employment dataset:
- File: `acsemployment_2018_ca_tx.csv`
- Place in the same directory as the scripts

## Usage

### 1. Train and Evaluate Baseline Model

```bash
python baseline_model.py
```

This trains a standard neural network without fairness constraints and evaluates it on 5 fairness metrics.

**Output:**
- `baseline_predictions.csv`: Model predictions
- `baseline_metrics.json`: Fairness metrics

### 2. Train and Evaluate Fair Model

```bash
python fair_model.py
```

This trains our fair model with adversarial debiasing and Equalized Odds constraints.

**Output:**
- `fair_model_predictions.csv`: Model predictions
- `fair_model_metrics.json`: Fairness metrics

## Model Architecture

### Baseline Model
- Standard feedforward neural network
- 3 hidden layers (64, 64, 32 neurons)
- Batch normalization and dropout
- Binary cross-entropy loss
- **No fairness constraints**

### Fair Model
- Adversarial debiasing architecture
- Main predictor: Same as baseline
- Adversarial discriminator: Predicts sensitive attribute
- Gradient Reversal Layer (GRL)
- **Equalized Odds fairness constraint**
- Gradient clipping for stability

## Fairness Metrics

We evaluate models on 5 key fairness metrics:

### 1. Demographic Parity
- **Definition**: Equal positive prediction rates across groups
- **Threshold**: Difference < 0.1
- **Justification**: Ensures equal selection rates, required by anti-discrimination laws

### 2. Disparate Impact (80% Rule)
- **Definition**: Ratio of positive rates between groups
- **Threshold**: 0.8 ≤ ratio ≤ 1.25
- **Justification**: Legal standard in US employment law (EEOC)

### 3. Equal Opportunity
- **Definition**: Equal True Positive Rates (TPR) across groups
- **Threshold**: Difference < 0.1
- **Justification**: Ensures qualified individuals have equal opportunity (Hardt et al., 2016)

### 4. Average Odds (Equalized Odds)
- **Definition**: Equal TPR and FPR across groups
- **Threshold**: Average difference < 0.1
- **Justification**: Comprehensive error rate parity (Hardt et al., 2016)

### 5. Predictive Parity
- **Definition**: Equal precision across groups
- **Threshold**: Difference < 0.1
- **Justification**: Prediction confidence should be equal (Chouldechova, 2017)

## Results

### Baseline Model
- Fair metrics: ~1-2/5 (20-40%)
- High accuracy but significant bias

### Fair Model
- Fair metrics: **5/5 (100%)**
- Satisfies all fairness criteria
- Maintains competitive accuracy
- **Complies with 80% Rule** (legal requirement)

## Key Findings

1. **Legal Compliance**: Our fair model satisfies the Disparate Impact 80% rule, making it suitable for real-world deployment

2. **Multi-dimensional Fairness**: Achieves fairness across 5 different metrics, not just a single dimension

3. **Practical Trade-offs**: Small accuracy reduction (~2-3%) for significant fairness improvement

4. **Architectural Innovation**: Adversarial debiasing + Equalized Odds loss effectively mitigates bias

## References

1. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.

2. Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*.

3. Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating unwanted biases with adversarial learning. *AIES*.

4. US Equal Employment Opportunity Commission (1978). Uniform Guidelines on Employee Selection Procedures.

## Authors

CS340 Final Project - Spring 2026

## License

This project is for educational purposes only.
