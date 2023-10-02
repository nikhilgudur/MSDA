import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# titanic = sns.load_dataset('titanic')

# passengers_age = titanic['age']

# np.sort(passengers_age)

# print(passengers_age)

# sns.ecdfplot(data=passengers_age.dropna())

# plt.plot()


df = pd.read_csv('./house_price_train.csv')
sale_price = df['SalePrice']


fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle("Sales and Log sales")


log_sales = np.log(df)
ax1.plot()

plt.title("Log")
sns.kdeplot(log_sales)
plt.subplot()
sns.set(style="darkgrid")
plt.show()


plt.title("Normal")
sns.kdeplot(df)

sns.set(style="darkgrid")

plt.show()
