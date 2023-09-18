import numpy as np
import pandas as pd
import statistics
from itertools import combinations_with_replacement
from scipy import stats
from scipy.special import comb


val = [82, 88, 76, 88, 92, 78, 89, 91, 85, 87,
       94, 75, 81, 90, 83, 86, 88, 92, 79, 84]

df = pd.DataFrame([82, 88, 76, 88, 92, 78, 89, 91, 85, 87,
                   94, 75, 81, 90, 83, 86, 88, 92, 79, 84])

print(df.describe())

# print("Mean", np.mean(val))

# print("Median", np.median(val))

# print("Mode", statistics.mode(val))

# # print(stats.mode(val))

# print("Standard Deviation", np.std(val))

# print("Variance", statistics.variance(val))

# q3, q1 = np.percentile(val, [75, 25])

# print("IQR", q3 - q1)

# print("scipy IQR", stats.iqr(val))

# print("Range", max(val) - min(val))

a = combinations_with_replacement(['B', 'R', 'G', 'Y'], 8)

i = 0

for c in a:
    i += 1

print('a', i)

# combination = comb(4, 8, exact=True, repetition=True)

# print(combination)
