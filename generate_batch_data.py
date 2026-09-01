import pandas as pd
import numpy as np

def generate_test_batch(num_records=100, output_path="test_batch.csv"):
    data = {
        "loan_amnt": np.random.uniform(1000, 40000, num_records).round(2),
        "term": np.random.choice([" 36 months", " 60 months"], num_records),
        "int_rate": np.random.uniform(5.0, 25.0, num_records).round(2),
        "installment": np.random.uniform(30.0, 1200.0, num_records).round(2),
        "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], num_records),
        "sub_grade": np.random.choice(["A1", "B2", "C3", "D4", "E5"], num_records),
        "emp_length": np.random.choice(["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"], num_records),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", "ANY"], num_records),
        "annual_inc": np.random.uniform(30000, 150000, num_records).round(2),
        "verification_status": np.random.choice(["Not Verified", "Source Verified", "Verified"], num_records),
        "purpose": np.random.choice(["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business"], num_records),
        "addr_state": np.random.choice(["CA", "NY", "TX", "FL", "IL"], num_records),
        "dti": np.random.uniform(1.0, 35.0, num_records).round(2),
        "delinq_2yrs": np.random.randint(0, 3, num_records).astype(float),
        "inq_last_6mths": np.random.randint(0, 4, num_records).astype(float),
        "open_acc": np.random.randint(2, 20, num_records).astype(float),
        "pub_rec": np.random.randint(0, 2, num_records).astype(float),
        "pub_rec_bankruptcies": np.random.randint(0, 2, num_records).astype(float),
        "revol_bal": np.random.uniform(0, 50000, num_records).round(2),
        "revol_util": np.random.uniform(0, 100, num_records).round(2),
        "total_acc": np.random.randint(5, 40, num_records).astype(float),
        "initial_list_status": np.random.choice(["w", "f"], num_records),
        "application_type": np.random.choice(["Individual", "Joint App"], num_records),
        "mort_acc": np.random.randint(0, 6, num_records).astype(float)
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_records} test records.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_test_batch(100)