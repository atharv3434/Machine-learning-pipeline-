"""
Generate a synthetic customer churn dataset for the PySpark MLlib pipeline.

This project ships with pre-generated data already in place
(data/customer_churn.csv), so you don't need to run this to try the
project out. Run it again for a fresh random sample, a different size, or
a different seed.

Usage:
    python data/generate_data.py [--n 5000] [--seed 42] [--out data/customer_churn.csv]
    
"""

import argparse
import numpy as np
import pandas as pd


def generate_customers(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, size=n)
    contract_type = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"], size=n, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(["Fiber", "DSL", "No Internet"], size=n, p=[0.45, 0.35, 0.20])
    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        size=n, p=[0.35, 0.2, 0.2, 0.25],
    )
    num_support_tickets = rng.poisson(1.2, size=n)
    age = rng.integers(18, 81, size=n)
    is_senior = (age >= 65).astype(int)

    base_charge = np.where(internet_service == "Fiber", 75, np.where(internet_service == "DSL", 50, 25))
    monthly_charges = base_charge + rng.normal(10, 12, size=n)
    monthly_charges = np.clip(monthly_charges, 20, 130)

    total_charges = tenure_months * monthly_charges + rng.normal(0, 50, size=n)
    total_charges = np.clip(total_charges, 0, None)

    contract_risk = np.where(contract_type == "Month-to-Month", 1.1, np.where(contract_type == "One Year", 0.2, -0.6))
    tech_support_protective = np.where(tech_support == "Yes", -0.5, 0.3)

    logit = (
        -1.6
        + contract_risk
        + tech_support_protective
        + 0.35 * num_support_tickets
        - 0.03 * tenure_months
        + 0.012 * (monthly_charges - 60)
        + 0.25 * is_senior
        + rng.normal(0, 0.5, size=n)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{10000 + i}" for i in range(n)],
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "num_support_tickets": num_support_tickets,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": np.round(total_charges, 2),
        "age": age,
        "is_senior": is_senior,
        "churn": churn,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customer churn data.")
    parser.add_argument("--n", type=int, default=5000, help="Number of customer records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default="data/customer_churn.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_customers(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    churn_rate = df["churn"].mean() * 100
    print(f"Wrote {len(df)} synthetic customer records to {args.out} (churn rate: {churn_rate:.1f}%)")


if __name__ == "__main__":
    main()