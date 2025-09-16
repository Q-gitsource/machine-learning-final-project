import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt     # Core plotting
import seaborn as sns               # Statistical plots
import numpy as np                  # Numerical operations
from scipy import stats             # Statistical Functions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score


# Download latest version
path = kagglehub.dataset_download("ashpalsingh1525/imdb-movies-dataset")
csv_path = path + "/" + os.listdir(str(path))[0]
housing_prices_dataset = pd.read_csv(csv_path)
untrimmed_df = pd.DataFrame(housing_prices_dataset)
movies_df = untrimmed_df[["release_date", "revenue", "runtime", "vote_average", "adult", "budget", "original_language", "genres", "keywords"]]
movies_df = movies_df.dropna()  # Drop rows missing values


# Hot encoder for categorical columns
encoder = OneHotEncoder()
categorical = ['adult', 'original_language', 'genres', 'keywords']
swapped = encoder.fit_transform(movies_df[categorical]).toarray()
feature_names = encoder.get_feature_names_out(categorical)
transdata = pd.DataFrame(swapped, columns= feature_names)
FinalMovies = pd.concat([movies_df.drop(categorical, axis=1).reset_index(drop=True), transdata.reset_index(drop=True)], axis=1)


# Separate features x and target y
x = FinalMovies.drop(["vote_average"], axis=1)
y = FinalMovies["vote_average"]


# train test split on data
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, test_size=0.8, random_state=53)
model = LinearRegression()
model.fit(x_tr,y_tr)
y_pred = model.predict(x_ts)
r2 = r2_score(y_ts, y_pred)


# Print Predictions and the Results
print("Predictions:", y_pred)
print("Actual:", y_ts.values)
print("R² Score:", r2)


## Visualization


# Set up
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots(figsize=(8, 8))

# Actual vs Predicted
ax.scatter(y_ts, y_pred, alpha=0.6, color='steelblue', s=50, label="Actual vs Predicted")

# Sort data for smooth line plotting
sort_idx = np.argsort(y_ts.flatten())
X_sorted = y_ts.flatten()[sort_idx]
y_pred_sorted = y_pred[sort_idx]

# Plot regression line
ax.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, 
        label=f'Regression line (R² = {r2:.3f})')

# Add equation text
slope = model.coef_
intercept = model.intercept_
equation = f'y = {slope:.3f}x + {intercept:.3f}'
ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        verticalalignment='top', fontsize=11)

# Formatting
ax.set_xlabel("Actual Vote Average")
ax.set_ylabel("Predicted Vote Average")
ax.set_title(f'Linear Regression Performance (R² = {r2:.3f})', 
            fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)



