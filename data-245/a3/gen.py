import numpy as np
import pandas as pd

# Generate 50 random samples with 2 feature columns
X = np.random.rand(50, 2)

# Generate class labels 0 and 1 randomly
y = np.random.randint(0, 2, size=50)

# Create dataframe from features and labels
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Class'] = y

# Export to CSV file
df.to_csv('dataset.csv', index=False)

print("CSV file generated with 50 examples, 2 features, and 2 class labels")
