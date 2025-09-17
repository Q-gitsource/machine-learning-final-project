import pandas as pd
import numpy as np
import kagglehub
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Download latest version of the dataset
path = kagglehub.dataset_download("ashpalsingh1525/imdb-movies-dataset")
csv_path = path + "/" + os.listdir(str(path))[0]
movies_df = pd.read_csv(csv_path)


# -----------------------------
# Preprocessing
# -----------------------------

# Simplifying broad column for more accurate data
def simplify_genre(genre):
    if pd.isna(genre):
        return "Other"
    if any(g in genre for g in ["Action", "Adventure", "Sci-Fi", "Fantasy"]):
        return "Action/Adventure"
    elif any(g in genre for g in ["Drama", "Romance", "Biography"]):
        return "Drama/Romance"
    elif "Comedy" in genre:
        return "Comedy"
    elif any(g in genre for g in ["Horror", "Thriller", "Mystery"]):
        return "Horror/Thriller"
    elif any(g in genre for g in ["Animation", "Family", "Musical"]):
        return "Family/Animation"
    elif any(g in genre for g in ["Crime", "War", "Western"]):
        return "Crime/War"
    else:
        return "Other"

movies_df["simple_genre"] = movies_df["genre"].apply(simplify_genre)

# # Log transform skewed numerical categories
# movies_df["budget_x"] = np.exp(movies_df["budget_x"] / 1000000000)# Values too large so divide by
# movies_df["revenue"] = np.exp(movies_df["revenue"] / 1000000000)

# Seperate categories
categorical = ["simple_genre", "orig_lang"]
numeric = ["budget_x", "revenue"]
target = "score"

# Drop missing / '0' values
movies_df = movies_df.dropna(subset=categorical + numeric + [target, "orig_title", "overview", "crew"])
movies_df = movies_df[
    (movies_df["budget_x"] > 0) & 
    (movies_df["revenue"] > 0)
    ]

# Encode categorical
encoder = OneHotEncoder(sparse_output=False)
cat_encoded = encoder.fit_transform(movies_df[categorical])
cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical))

# TF-IDF (title + overview combined)
movies_df["text"] = movies_df["orig_title"].astype(str) + " " + movies_df["overview"].astype(str)
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
tfidf = vectorizer.fit_transform(movies_df["text"])
svd = TruncatedSVD(n_components=50, random_state=53)
text_reduced = svd.fit_transform(tfidf)
text_df = pd.DataFrame(text_reduced, columns=[f"tfidf_text_{i}" for i in range(text_reduced.shape[1])])

# Convert date_x to year month
movies_df["year"] = pd.to_datetime(movies_df["date_x"]).dt.year
movies_df["month"] = pd.to_datetime(movies_df["year"]).dt.month
movies_df = movies_df.drop(["date_x"], axis=1)

# TF-IDF (crew)
movies_df["crew"] = movies_df["crew"].astype(str)
crew_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
crew_tfidf = crew_vectorizer.fit_transform(movies_df["crew"])
crew_svd = TruncatedSVD(n_components=100, random_state=53)
crew_reduced = crew_svd.fit_transform(crew_tfidf)
crew_df = pd.DataFrame(crew_reduced, columns=[f"tfidf_crew_{i}" for i in range(crew_reduced.shape[1])])


# Final dataset
FinalMovies = pd.concat([
    movies_df[numeric].reset_index(drop=True),
    cat_df.reset_index(drop=True),
    text_df.reset_index(drop=True),
    crew_df.reset_index(drop=True)
], axis=1)

y = movies_df[target].values

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(FinalMovies, y, test_size=0.2, random_state=53)

# -----------------------------
# Models
# -----------------------------
# Define feature subsets
rf_features = numeric + list(cat_df.columns)
ridge_features = [c for c in FinalMovies.columns if "tfidf_" in c]

ridge_features = [c for c in FinalMovies.columns if "tfidf_" in c]
param_grid = {'alpha': [i for i in range(1,10)]}
ridge_grid = GridSearchCV(
    Ridge(),
    param_grid,
    scoring='r2',   # optimize R²
    cv=5,
    n_jobs=-1
)
ridge_grid.fit(X_train[ridge_features], y_train)

# Best alpha
best_alpha = ridge_grid.best_params_['alpha']
print("Best alpha:", best_alpha)

# Transformers to select subsets
rf_selector = ColumnTransformer([("select", "passthrough", rf_features)], remainder="drop")
ridge_selector = ColumnTransformer([("select", "passthrough", ridge_features)], remainder="drop")

# Pipelines for each base estimator
rf_pipeline = Pipeline([
    ("selector", rf_selector),
    ("rf", RandomForestRegressor(n_estimators=400, max_depth=None, random_state=53, n_jobs=-1))
])

ridge_pipeline = Pipeline([
    ("selector", ridge_selector),
    ("ridge", Ridge(alpha=best_alpha))
])

# Stacking regressor with both pipelines
stack = StackingRegressor(
    estimators=[
        ('rf', rf_pipeline),
        ('ridge', ridge_pipeline)
    ],
    final_estimator=Ridge(),  # meta-model learns optimal weights
    n_jobs=-1
)

# -----------------------------
# Fit stacking model
# -----------------------------
stack.fit(X_train, y_train)

# Predict
y_pred_stack = stack.predict(X_test)
print("R²:", r2_score(y_test, y_pred_stack))

# # Ensemble
# y_pred = (rf_pred + ridge_pred) / 2
# print("R²:", r2_score(y_test, y_pred))


# -----------------------------
# Visualisation
# -----------------------------

## Predicted vs Actual
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_stack, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Predicted vs Actual Movie Scores")
plt.show()

## Feature Importance
rf_fitted = stack.named_estimators_['rf'].named_steps['rf']
importances = rf_fitted.feature_importances_
indices = np.argsort(importances)[-20:]  # top 20

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [FinalMovies.columns[i] for i in indices])
plt.title("Top Feature Importances (RandomForest)")
plt.show()

# Residuals
residuals = y_test - y_pred_stack
plt.figure(figsize=(6,4))
plt.scatter(y_pred_stack, residuals, alpha=0.4)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Distribution check
plt.hist(y_test, bins=30, alpha=0.5, label="Actual")
plt.hist(y_pred_stack, bins=30, alpha=0.5, label="Predicted")
plt.legend()
plt.title("Distribution of Actual vs Predicted Scores")
plt.show()


# -----------------------------
# Revenue vs Score
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(movies_df["revenue"], movies_df["score"], alpha=0.3)
plt.xscale("log")
plt.xlabel("Revenue")
plt.ylabel("Vote Average (Score)")
plt.title("Revenue vs Vote Average")
plt.show()

# -----------------------------
# Budget vs Score
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(movies_df["budget_x"], movies_df["score"], alpha=0.3, color="orange")
plt.xscale("log")
plt.xlabel("Budget")
plt.ylabel("Vote Average (Score)")
plt.title("Budget vs Vote Average")
plt.show()
