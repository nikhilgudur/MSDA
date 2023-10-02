# Seaborn practice

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure(figsize=(6, 8))

x = np.linspace(0, 10, 100)
y = np.sin(x)

# themes = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

# for i, theme in enumerate(themes):
#     with sns.axes_style(theme):
#         sns.lineplot(x=x, y=y)


# plt.tight_layout()
# plt.show()

categories = ["Math", "English", "History", "Art"]
values = [85, 92, 78, 88, 95]

plt.figure(figsize=(6, 2))
sns.barplot(x=categories, y=values)
plt.show()
