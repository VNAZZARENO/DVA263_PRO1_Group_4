import pandas as pd
import numpy as np
import os
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from data.utils import MixtureOfExperts, model_pipeline


def preprocess_data(output_dir, datasets):
    preprocessed_data = {}

    for name, data in datasets.items():
        print(f"Processing {name}...")
        data.fillna(data.mean(numeric_only=True), inplace=True)  
        try:
            data.drop(columns="Unnamed: 0", axis=1, inplace=True)
        except:
            pass
        preprocessed_data[name] = data

    for name, data in preprocessed_data.items():
        output_path = os.path.join(output_dir, f'{name}_preprocessed.csv')
        data.to_csv(output_path, index=False)

    print("Preprocessing complete. Files saved:")
    print(", ".join([os.path.join(output_dir, f"{name}_preprocessed.csv") for name in preprocessed_data]))


def create_features(df):
    df_processed = df.copy()
    # new features from existing ones from research on dataset
    df_processed.drop_duplicates(inplace=True)
    return df_processed
