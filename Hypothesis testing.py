import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency, shapiro, ks_1samp

# Dummy Data Generation
np.random.seed(0)

# Population Parameters
population_mean = 170
population_std = 10

# Generate a large sample for Z-Test
large_sample = np.random.normal(population_mean, population_std, 100)

# Generate a small sample for T-Tests
small_sample = np.random.normal(172, 8, 25)

# Two independent samples for Two-Sample T-Test
group1 = np.random.normal(172, 8, 30)
group2 = np.random.normal(170, 7, 30)

# Paired data for Paired T-Test
before_treatment = np.random.normal(170, 5, 20)
after_treatment = before_treatment + np.random.normal(1, 2, 20)

# Generate data for ANOVA
group_A = np.random.normal(172, 5, 30)
group_B = np.random.normal(175, 5, 30)
group_C = np.random.normal(168, 5, 30)

# Generate data for Chi-Square Test (categorical)
observed = np.array([[10, 20, 30], [6, 18, 36]])

# Z-Test (large sample, known variance)
z_statistic, z_p_value = stats.ttest_1samp(large_sample, population_mean)
print(f"Z-Test:\nZ-Statistic: {z_statistic}, P-Value: {z_p_value}\n")

# One-Sample T-Test (small sample, unknown variance)
t_statistic, t_p_value = stats.ttest_1samp(small_sample, population_mean)
print(f"One-Sample T-Test:\nT-Statistic: {t_statistic}, P-Value: {t_p_value}\n")

# Two-Sample T-Test (independent groups)
t2_statistic, t2_p_value = stats.ttest_ind(group1, group2)
print(f"Two-Sample T-Test:\nT-Statistic: {t2_statistic}, P-Value: {t2_p_value}\n")

# Paired T-Test (same group before and after treatment)
paired_t_statistic, paired_t_p_value = stats.ttest_rel(before_treatment, after_treatment)
print(f"Paired T-Test:\nT-Statistic: {paired_t_statistic}, P-Value: {paired_t_p_value}\n")

# ANOVA (compare 3 groups)
df = pd.DataFrame({
    'score': np.concatenate([group_A, group_B, group_C]),
    'group': ['A'] * 30 + ['B'] * 30 + ['C'] * 30
})

anova_model = ols('score ~ group', data=df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(f"ANOVA:\n{anova_table}\n")

# Chi-Square Test (categorical data)
chi2_stat, chi2_p_value, dof, expected = chi2_contingency(observed)
print(f"Chi-Square Test:\nChi-Square Statistic: {chi2_stat}, P-Value: {chi2_p_value}\n")

# Shapiro-Wilk Test (normality test for small sample)
shapiro_stat, shapiro_p_value = shapiro(small_sample)
print(f"Shapiro-Wilk Test:\nShapiro Statistic: {shapiro_stat}, P-Value: {shapiro_p_value}\n")

# Kolmogorov-Smirnov Test (normality test for larger sample)
ks_stat, ks_p_value = ks_1samp(large_sample, 'norm', args=(population_mean, population_std))
print(f"Kolmogorov-Smirnov Test:\nKS Statistic: {ks_stat}, P-Value: {ks_p_value}\n")
