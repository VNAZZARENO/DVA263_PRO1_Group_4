# preprocessing.py
import pandas as pd
import numpy as np
import os
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from data.utils import MixtureOfExperts, model_pipeline

from sklearn.decomposition import PCA


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
    
    # violation_cols = [
    #     'violation_no', 'violation_roadside_invasion', 'violation_give_the_way',
    #     'violation_road_signs', 'violation_heavy_braking', 'violation_roadside_exit',
    #     'violation_slowdown'
    # ]
    # df_processed["total_violations"] = df_processed[violation_cols].sum(axis=1)
    
    # emotion_cols = [
    #     'emotion_na', 'emotion_neutral', 'emotion_surprise',
    #     'emotion_anger', 'emotion_boredom', 'emotion_fear', 'emotion_fun'
    # ]
    # df_processed["num_emotions"] = df_processed[emotion_cols].sum(axis=1)

    # distraction_cols = [
    #     'distraction_na', 'distraction_no', 'distraction_reduced_attention',
    #     'distraction_speaking'
    # ]
    # df_processed["num_distractions"] = df_processed[distraction_cols].sum(axis=1)
    
    
    # def map_gaze(row):
    #     if row["gaze_na"] == 0.0: 
    #         return 0
    #     elif row["gaze_poor_expl"] == 0.5:
    #         return 1
    #     elif row["gaze_partial_expl"] == 1.0:
    #         return 2
    #     elif row["gaze_complete_expl"] == 1.5:
    #         return 3
    #     return 0  

    # gaze_cols = ['gaze_na','gaze_poor_expl','gaze_partial_expl','gaze_complete_expl']
    # df_processed["gaze_level"] = df_processed[gaze_cols].apply(map_gaze, axis=1)

    # df_processed["speed_ratio"] = df_processed["avg_speed"] / (df_processed["max_speed"] + 1e-9)

    # df_processed["acce_magnitude"] = (
    #     df_processed["lat_acce"]**2 + df_processed["long_acce"]**2
    # )**0.5

    # df_processed["beta_alpha_ratio_frontal"] = (
    #     df_processed["BetaFrontal"] / (df_processed["AlphaFrontal"] + 1e-9)
    # )
    # df_processed["beta_alpha_ratio_parietal"] = (
    #     df_processed["BetaParietal"] / (df_processed["AlphaParietal"] + 1e-9)
    # )

    # df_processed.drop_duplicates(inplace=True)

    return df_processed



def apply_pca(data, n_components=None):
    """
    Apply Principal Component Analysis (PCA) to reduce the dimensions of the dataset.
    
    Parameters:
    - data: pandas DataFrame or numpy array
        The dataset to be transformed.
    - n_components: int or None
        Number of components to keep. If None, all components are kept.
    
    Returns:
    - transformed_data: numpy array
        The dataset after PCA transformation.
    - pca: PCA object
        The fitted PCA object, which can be used for further analysis.
    """

    data_numeric = data.select_dtypes(include=[np.number])
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data_numeric)
    return transformed_data, pca