"""
Compare baseline and fair model results on extended fairness metrics.
"""

import json
import pandas as pd

def load_metrics(filename):
    """Load metrics from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def print_comparison():
    """Print comparison of baseline and fair model."""

    print("="*80)
    print("  BASELINE vs FAIR MODEL COMPARISON")
    print("="*80)

    # Load metrics
    baseline = load_metrics('baseline_metrics.json')
    fair = load_metrics('fair_model_metrics.json')

    # Print header
    print("\n{:<25} {:<15} {:<15} {:<10}".format(
        "Metric", "Baseline", "Fair Model", "Status"
    ))
    print("-"*80)

    # 1. Demographic Parity
    b_dp = baseline['demographic_parity']['difference']
    f_dp = fair['demographic_parity']['difference']
    b_dp_fair = baseline['demographic_parity']['fair']
    f_dp_fair = fair['demographic_parity']['fair']

    status = "✓" if f_dp_fair and not b_dp_fair else ("=" if f_dp_fair == b_dp_fair else "✗")
    print("{:<25} {:.4f} {:>6}    {:.4f} {:>6}    {}".format(
        "Demographic Parity",
        b_dp, "✓" if b_dp_fair else "✗",
        f_dp, "✓" if f_dp_fair else "✗",
        status
    ))

    # 2. Disparate Impact
    b_di = baseline['disparate_impact']['ratio']
    f_di = fair['disparate_impact']['ratio']
    b_di_fair = baseline['disparate_impact']['fair']
    f_di_fair = fair['disparate_impact']['fair']

    status = "✓" if f_di_fair and not b_di_fair else ("=" if f_di_fair == b_di_fair else "✗")
    print("{:<25} {:.4f} {:>6}    {:.4f} {:>6}    {}".format(
        "Disparate Impact",
        b_di, "✓" if b_di_fair else "✗",
        f_di, "✓" if f_di_fair else "✗",
        status
    ))

    # 3. Equal Opportunity
    b_eo = baseline['equal_opportunity']['difference']
    f_eo = fair['equal_opportunity']['difference']
    b_eo_fair = baseline['equal_opportunity']['fair']
    f_eo_fair = fair['equal_opportunity']['fair']

    status = "✓" if f_eo_fair and not b_eo_fair else ("=" if f_eo_fair == b_eo_fair else "✗")
    print("{:<25} {:.4f} {:>6}    {:.4f} {:>6}    {}".format(
        "Equal Opportunity",
        b_eo, "✓" if b_eo_fair else "✗",
        f_eo, "✓" if f_eo_fair else "✗",
        status
    ))

    # 4. Average Odds
    b_ao = baseline['average_odds']['difference']
    f_ao = fair['average_odds']['difference']
    b_ao_fair = baseline['average_odds']['fair']
    f_ao_fair = fair['average_odds']['fair']

    status = "✓" if f_ao_fair and not b_ao_fair else ("=" if f_ao_fair == b_ao_fair else "✗")
    print("{:<25} {:.4f} {:>6}    {:.4f} {:>6}    {}".format(
        "Average Odds",
        b_ao, "✓" if b_ao_fair else "✗",
        f_ao, "✓" if f_ao_fair else "✗",
        status
    ))

    # 5. Predictive Parity
    b_pp = baseline['predictive_parity']['difference']
    f_pp = fair['predictive_parity']['difference']
    b_pp_fair = baseline['predictive_parity']['fair']
    f_pp_fair = fair['predictive_parity']['fair']

    status = "✓" if f_pp_fair and not b_pp_fair else ("=" if f_pp_fair == b_pp_fair else "✗")
    print("{:<25} {:.4f} {:>6}    {:.4f} {:>6}    {}".format(
        "Predictive Parity",
        b_pp, "✓" if b_pp_fair else "✗",
        f_pp, "✓" if f_pp_fair else "✗",
        status
    ))

    print("-"*80)

    # Summary
    b_score = baseline['summary']['fair_metrics']
    f_score = fair['summary']['fair_metrics']

    print("\n{:<25} {:<15} {:<15} {:<10}".format(
        "SUMMARY", "Baseline", "Fair Model", "Improvement"
    ))
    print("-"*80)
    print("{:<25} /5 ({:.0%})      {}/5 ({:.0%})      {}".format(
        "Fair Metrics",
        b_score, b_score/5,
        f_score, f_score/5,
        f"+{f_score - b_score}" if f_score > b_score else ("=" if f_score == b_score else f"{f_score - b_score}")
    ))

    print("="*80)

    # Detailed improvements
    improvements = []
    if f_dp_fair and not b_dp_fair:
        improvements.append("Demographic Parity")
    if f_di_fair and not b_di_fair:
        improvements.append("Disparate Impact")
    if f_eo_fair and not b_eo_fair:
        improvements.append("Equal Opportunity")
    if f_ao_fair and not b_ao_fair:
        improvements.append("Average Odds")
    if f_pp_fair and not b_pp_fair:
        improvements.append("Predictive Parity")

    if improvements:
        print("\n✓ IMPROVED METRICS:")
        for metric in improvements:
            print(f"  - {metric}")
    else:
        print("\n= NO IMPROVEMENTS (both models have same fairness score)")

    print("\n")

if __name__ == "__main__":
    print_comparison()
