import pandas as pd
import numpy as np

np.random.seed(42)  # Ensure reproducibility

# Generate dataset
credit_scores = np.random.randint(300, 850, size=150000)
annual_incomes = np.random.randint(10000, 100000, size=150000)
debt_to_income_ratios = np.random.uniform(0, 1, size=150000)
number_of_open_accounts = np.random.randint(1, 30, size=150000)
loan_amounts = np.random.randint(500, 50000, size=150000)
loan_terms = np.random.choice([15, 30], size=150000)
previous_defaults = np.random.choice([0, 1], size=150000)
defaults = np.random.choice([0, 1], size=150000)

# Create DataFrame
df = pd.DataFrame({
    'CreditScore': credit_scores,
    'AnnualIncome': annual_incomes,
    'DebtToIncomeRatio': debt_to_income_ratios,
    'NumberOfOpenAccounts': number_of_open_accounts,
    'LoanAmount': loan_amounts,
    'LoanTerm': loan_terms,
    'PreviousDefaults': previous_defaults,
    'Default': defaults
})

# Save the DataFrame to a CSV file
df.to_csv('loan_default_prediction_150k.csv', index=False)

print("Dataset generated and saved to 'loan_default_prediction_dataset.csv'.")
