import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

# Create a DataFrame
df = pd.DataFrame(data, columns=iris.feature_names)
df['target'] = target

# Save it to a CSV file
df.to_csv('data.csv', index=False)

print("Dataset saved as data.csv")