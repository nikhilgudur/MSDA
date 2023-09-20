import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")

print(penguins)

sns.kdeplot(penguins["body_mass_g"])
sns.set_theme("darkgrid")

plt.show()
