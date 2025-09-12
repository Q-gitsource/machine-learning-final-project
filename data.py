import os
import kagglehub
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Download latest version
path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")

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

x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, test_size=0.8)

log = LogisticRegression()
log.fit(x_tr,y_tr)
