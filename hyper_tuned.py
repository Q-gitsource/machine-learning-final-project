import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt     # Core plotting
import numpy as np                  # Numerical operations
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Download latest version of the dataset
path = kagglehub.dataset_download("ashpalsingh1525/imdb-movies-dataset")
csv_path = path + "/" + os.listdir(str(path))[0]
untrimmed_df = pd.read_csv(csv_path)
features = ["budget_x", "score", "genre", "orig_lang", "date_x", "orig_title", "overview"]

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

# Simplifying broad column for more accurate data
def simplify_genre(genre):
    if "Action" in genre or "Adventure" in genre or "Sci-Fi" in genre or "Fantasy" in genre:
        return "Action/Adventure"
    elif "Drama" in genre or "Romance" in genre or "Biography" in genre:
        return "Drama/Romance"
    elif "Comedy" in genre:
        return "Comedy"
    elif "Horror" in genre or "Thriller" in genre or "Mystery" in genre:
        return "Horror/Thriller"
    elif "Animation" in genre or "Family" in genre or "Musical" in genre:
        return "Family/Animation"
    elif "Crime" in genre or "War" in genre or "Western" in genre:
        return "Crime/War"
    else:
        return "Other"


movies_df["genre"] = movies_df["genre"].apply(simplify_genre)


# Hot encoder for categorical columns
encoder = OneHotEncoder()
categorical = ["genre", "orig_lang"]
swapped = encoder.fit_transform(movies_df[categorical]).toarray()
feature_names = encoder.get_feature_names_out(categorical)
transdata = pd.DataFrame(swapped, columns= feature_names)
FinalMovies = pd.concat([movies_df.drop(categorical, axis=1).reset_index(drop=True), transdata.reset_index(drop=True)], axis=1)

## Data Optimization
# Convert date_x to year month
FinalMovies["year"] = pd.to_datetime(FinalMovies["date_x"]).dt.year
FinalMovies["month"] = pd.to_datetime(FinalMovies["year"]).dt.month
FinalMovies = FinalMovies.drop(["date_x"], axis=1)

# Term Frequency -- Inverse Document Frequency Vectorizer
overview_vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
orig_title_vectorizer = TfidfVectorizer(stop_words="english", max_features=200)

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

# Log transform skewed budget
FinalMovies["budget_x"] = np.log1p(FinalMovies["budget_x"])

## Train model on data
# Separate features x and target y
x = FinalMovies.drop(["score"], axis=1)
y = FinalMovies["score"]

# standardization for performance boost

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# train test split on data
x_tr, x_ts, y_tr, y_ts = train_test_split(x_scaled, y, test_size=0.2, random_state=53)
params = {"C": [0.1, 1, 10], "epsilon": [0.01, 0.1, 1], "kernel": ["linear", "rbf"]}
model = GridSearchCV(SVR(), params, cv=5, scoring="r2", n_jobs=-1)
model.fit(x_tr, y_tr)
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
x_feat_scaled = scaler.transform(x_feature)
# Predict with model
y_feature_pred = model.predict(x_feat_scaled)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(x[feature], y, alpha=0.3, label="Actual Data", s=20)
plt.plot(feature_range, y_feature_pred, color="red", linewidth=2, label="Model Prediction")
plt.xlabel(feature.capitalize())
plt.ylabel("Vote Average")
plt.title(f"Effect of {feature.capitalize()} on Predicted Vote Average")
plt.legend()
#plt.show()

# r2_score of 5.7%
# r2_score after standardization 6.5%
# r2_score after standard + simple genres 14%


# -----------------------------
# Visualisation
# -----------------------------

## Predicted vs Actual
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Predicted vs Actual Movie Scores")
plt.show()

## Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[-20:]  # top 20

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [FinalMovies.columns[i] for i in indices])
plt.title("Top Feature Importances (RandomForest)")
plt.show()