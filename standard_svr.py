import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt     # Core plotting
import numpy as np                  # Numerical operations
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Download latest version of the dataset
path = kagglehub.dataset_download("ashpalsingh1525/imdb-movies-dataset")
csv_path = path + "/" + os.listdir(str(path))[0]
untrimmed_df = pd.read_csv(csv_path)
features = ["budget_x", "score", "genre", "orig_lang", "date_x", "orig_title", "overview", "revenue"]

## Data Filtering
# Select features to analyze
movies_df = untrimmed_df[features]

# Size before and after dropping missing values
before_na = len(movies_df)
movies_df = movies_df.dropna()  # Drop rows missing values
after_na = len(movies_df)
print(f"{before_na} rows before, {after_na} rows after, {before_na - after_na} removed due to missing values")

# Remove entries with 0 in key numeric columns
before_filter = len(movies_df)
movies_df = movies_df[
    (movies_df["budget_x"] > 0) &
    (movies_df["score"] > 0)
]
after_filter = len(movies_df)
print(f"{before_filter} rows before, {after_filter} rows after, {before_filter - after_filter} removed due to '0' values")

# Hot encoder for categorical columns
encoder = OneHotEncoder()
categorical = ["genre", "orig_lang"]
swapped = encoder.fit_transform(movies_df[categorical]).toarray()
feature_names = encoder.get_feature_names_out(categorical)
transdata = pd.DataFrame(swapped, columns= feature_names)
FinalMovies = pd.concat([movies_df.drop(categorical, axis=1).reset_index(drop=True), transdata.reset_index(drop=True)], axis=1)

## Data Optimization
# Convert date_x to year
FinalMovies["year"] = pd.to_datetime(FinalMovies["date_x"]).dt.year
FinalMovies = FinalMovies.drop(["date_x"], axis=1)

# Term Frequency -- Inverse Document Frequency Vectorizer
overview_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
orig_title_vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)

# Transform each Text Column
overview_tfidf = overview_vectorizer.fit_transform(FinalMovies["overview"].astype(str))
orig_title_tfidf = orig_title_vectorizer.fit_transform(FinalMovies["orig_title"].astype(str))

# Convert to Dataframes
overview_df = pd.DataFrame(overview_tfidf.toarray(),
                           columns=[f"overview_{w}" for w in overview_vectorizer.get_feature_names_out()])
orig_title_df = pd.DataFrame(orig_title_tfidf.toarray(),
                           columns=[f"title_{w}" for w in orig_title_vectorizer.get_feature_names_out()])

FinalMovies = pd.concat(
    [
        FinalMovies.drop(columns=["overview", "orig_title"]).reset_index(drop=True),
        overview_df.reset_index(drop=True),
        orig_title_df.reset_index(drop=True)
    ],
    axis=1
)

## Train model on data
# Separate features x and target y
x = FinalMovies.drop(["score"], axis=1)
y = FinalMovies["score"]

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
feature = "budget_x"

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