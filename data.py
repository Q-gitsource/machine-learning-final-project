import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Download latest version
path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")

csv_path = path + "/" + os.listdir(str(path))[0]

housing_prices_dataset = pd.read_csv(csv_path)
untrimmed_df = pd.DataFrame(housing_prices_dataset)
movies_df = untrimmed_df[["title", "release_date", "revenue", "runtime", "vote_average", "adult", "budget", "original_language", "genres", "keywords"]]
movies_df.dropna(inplace=True) # Drop rows missing values

# Separate features x and target y
x = movies_df.drop("vote_average", axis=1)
encoder = OneHotEncoder()
x_encoded = encoder.fit_transform(x)

y = movies_df["vote_average"]

print(x_encoded)