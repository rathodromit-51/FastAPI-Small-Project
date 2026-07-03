from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
housing = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(
    housing.data, # type: ignore
    columns=housing.feature_names # type: ignore
)

# Add target column
df["Price"] = housing.target # type: ignore

print("Shape : ", df.shape)
print(df.head())
# print(df.describe())