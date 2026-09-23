# Beta-Returns-Calculation-and-using-Python-
An analysis of the uploaded Jupyter Notebook (.ipynb) indicates that it focuses on calculating the financial Beta ($\beta$) of Alphabet Inc. (GOOGL) relative to the S&amp;P 500 (^GSPC) index using daily returns from January 2023 to January 2026.
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Recreating the returns DataFrame based on the notebook execution outputs
dates = ['2023-01-04', '2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10']
r_googl = [-0.011670, -0.021344, 0.013225, 0.007786, 0.004545]
r_sp500 = [0.007539, -0.011646, 0.022841, -0.000768, 0.006978]

returns_df = pd.DataFrame({'r_googl': r_googl, 'r_sp500': r_sp500}, index=dates)

# Step-by-step arithmetic verification:
# 1. Calculate sample means
mean_googl = np.mean(r_googl)
mean_sp500 = np.mean(r_sp500)

# 2. Deviations from the mean
deviations_googl = returns_df['r_googl'] - mean_googl
deviations_sp500 = returns_df['r_sp500'] - mean_sp500

# 3. Product of deviations and sample covariance
product_deviations = deviations_googl * deviations_sp500
cov_googl_sp500 = product_deviations.sum() / (len(returns_df) - 1)

# 4. Sample variance of S&P 500
var_sp500 = np.var(returns_df['r_sp500'], ddof=1)

# 5. Beta calculation
beta_googl = cov_googl_sp500 / var_sp500

# 6. Linear Regression check
reg_result = linregress(x=returns_df['r_sp500'], y=returns_df['r_googl'])
