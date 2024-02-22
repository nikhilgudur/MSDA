import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")

print(df.head())

print(df.isnull().sum())

print(df.describe())

cross_tab = pd.crosstab(df['admit'], df['rank'])
print(cross_tab)


logit = sm.Logit(df['admit'], df[['gre', 'gpa', 'rank']])

result = logit.fit()

print(result.summary())

x = pd.DataFrame({'gre': [790], 'gpa': [3.8], 'rank': [1]})
x['rank'] = x['rank'].astype('category')

predicted_probabilities = result.predict(x)

print(predicted_probabilities)
