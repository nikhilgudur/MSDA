import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('./icw/house_price_train.csv')

sales = df['SalePrice']

print("Sales mean", np.mean(sales))
print("Sales median", np.median(sales))
print("Sales mode", stats.mode(sales))


log = np.log(sales)

print("Log mean", np.mean(log))
print("Log median", np.median(log))
print("Log mode", stats.mode(log))
