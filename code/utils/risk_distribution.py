##############################
# Generate risk distribution #
##############################
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path
import os

# Make a new directory
PATH = Path('...')
SAVE_PATH = PATH / 'data/processed/a_risk/'
os.makedirs(SAVE_PATH, exist_ok=True)

# Parameters
variance = 0.68
sd = np.sqrt(variance)
pop_mean = -(variance/2)
case_mean = variance/2

# 10 year absolute risk
df = pd.read_excel(f'{PATH}/data/processed/devcan_absoluterisk.xlsx', header=2)
df = df.rename(columns=({'90+': 91}))

# Get the appropriate 10-year absolute risk value from devcan
absrisk = []
for row in np.arange(0, 81):
    absrisk.append(df.loc[row, row+10])

AR = pd.DataFrame(absrisk)
AR = AR.reset_index().rename(columns={'index': 'age', 0: '10yr_AR'})
AR = AR[45:80]

# Create reference absolute risks to compare each age against
reference_AR = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055,
                0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]

# Calculate RR by age, using 10 year AR threshold of that of the reference year
for value in reference_AR:

    rr = np.log(1-value) / np.log(1-AR['10yr_AR'])
    log_rr = np.log(rr)
    p_above_threshold = 1-norm.cdf(log_rr, pop_mean, sd)
    p_not_above_threshold = 1-p_above_threshold
    p_case = 1-norm.cdf(log_rr, case_mean, sd)
    p_noncase = 1-p_case
    rr_low = p_noncase / p_not_above_threshold
    rr_high = p_case / p_above_threshold

    a_risk = pd.DataFrame({
        'age': np.arange(45, 80),
        '10yr_AR': AR['10yr_AR'],
        'rr': rr,
        'log_rr': log_rr,
        'p_above_threshold': p_above_threshold,
        'p_case': p_case,
        'rr_low': rr_low,
        'rr_high': rr_high
    })

    a_risk.to_csv(f'{SAVE_PATH}/a_risk_{str(np.round(value*100, 2))}.csv')
