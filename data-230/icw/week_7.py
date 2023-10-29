import seaborn as sns
import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

df = sns.load_dataset('mpg')

print(df.head())

plt.title("Horsepower")
plt.figure(figsize=(10, 5))
sns.boxenplot(x='origin', y='horsepower', data=df)
plt.show()
