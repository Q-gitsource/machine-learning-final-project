import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt     # Core plotting
import seaborn as sns               # Statistical plots
import numpy as np                  # Numerical operations
from scipy import stats             # Statistical Functions
# from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# Download latest version of the dataset
path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
csv_path = path + "/" + os.listdir(str(path))[0]
untrimmed_df = pd.read_csv(csv_path)
features = ["revenue", "runtime", "vote_average", "vote_count", "budget"]

## Data Filtering
# Select features to analyze
movies_df = untrimmed_df[features]

# Size before and after dropping missing values
before_na = len(movies_df)
movies_df = movies_df.dropna()  # Drop rows missing values
after_na = len(movies_df)
print(f"{before_na} rows before, {after_na} rows after, {before_na - after_na} removed due to missing values")

# Size before and after dropping 0 values
before_filter = len(movies_df)
movies_df = movies_df[
    (movies_df["vote_count"] > 0) &
    (movies_df["budget"] > 0) &
    (movies_df["revenue"] > 0) &
    (movies_df["runtime"] > 0)
]
after_filter = len(movies_df)
print(f"{before_filter} rows before, {after_filter} rows after, {before_filter - after_filter} removed due to '0' values")

## Train model on data
# Separate features x and target y
x = movies_df.drop(["vote_average"], axis=1)
y = movies_df["vote_average"]

# train test split on data
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, test_size=0.8, random_state=53)
model = SVR()
model.fit(x_tr,y_tr)
y_pred = model.predict(x_ts)
r2 = r2_score(y_ts, y_pred)

# Print Predictions and the Results
print("Predictions:", y_pred)
print("Actual:", y_ts.values)
print("R² Score:", r2)


## Visualization
# Choose the feature you want to isolate
feature = "revenue"

# Copy test set to keep structure
x_mean = x.mean()   # mean of each feature
feature_range = np.linspace(x[feature].min(), x[feature].max(), 200)  # smooth range

# Construct dataset: all features = mean, except varying 'budget'
x_feature = pd.DataFrame([x_mean] * len(feature_range))
x_feature[feature] = feature_range

# Predict with model
y_feature_pred = model.predict(x_feature)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(x[feature], y, alpha=0.3, label="Actual Data", s=20)
plt.plot(feature_range, y_feature_pred, color="red", linewidth=2, label="Model Prediction")
plt.xlabel(feature.capitalize())
plt.ylabel("Vote Average")
plt.title(f"Effect of {feature.capitalize()} on Predicted Vote Average")
plt.legend()
plt.show()