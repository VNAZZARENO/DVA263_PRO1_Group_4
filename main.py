# main.py
import os
import logging
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats

from data.utils import model_pipeline
from data.preprocessing import preprocess_data, create_features

logging.basicConfig(level=logging.INFO)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    about_data = pd.read_excel(os.path.join(args.data_dir, 'About_the_dataset.xlsx'))
    simulation_data = pd.read_excel(os.path.join(args.data_dir, 'Feature_Simulation.xlsx'))
    track_data = pd.read_excel(os.path.join(args.data_dir, 'Feature_Track.xlsx'))

    datasets = {
        "about_data": about_data,
        "simulation_data": simulation_data,
        "track_data": track_data,
    }

    preprocess_data(args.output_dir, datasets)

    df_merged = pd.merge(
        simulation_data, 
        track_data, 
        on="subject",   
        how="inner"            
    )
    for col in df_merged.columns:
        if col.endswith('_x'):
            base_col = col[:-2]  
            if f"{base_col}_y" in df_merged.columns:
                df_merged[base_col] = df_merged[[col, f"{base_col}_y"]].mean(axis=1)
                df_merged.drop(columns=[col, f"{base_col}_y"], inplace=True)

    df_merged = create_features(df_merged)
    df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)

    results = model_pipeline(
        df=df_merged, 
        target=args.target_column, 
        task_type=args.task_type, 
        subset_frac=args.subset_frac, 
        random_state=args.random_state
    )

    model_names = []
    best_scores = []
    best_params_list = []
    mse_scores = []
    r2_scores = []
    y_pred_list = []

    for result in results:
        y_test = result['y_test']
        y_pred = result['y_pred']
        
        model_names.append(result['model'])
        best_scores.append(result['best_score'])
        best_params_list.append(result['best_params'])
        y_pred_list.append(y_pred)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        r2_scores.append(r2)

    num_models = len(results)
    fig, axes = plt.subplots(num_models, 2, figsize=(12, 4 * num_models), squeeze=False)

    os.makedirs("results", exist_ok=True)

    for i in range(num_models):
        y_pred = y_pred_list[i]
        residuals = y_test - y_pred
        
        axes[i, 0].scatter(y_test, y_pred, color='blue', alpha=0.5)
        axes[i, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes[i, 0].set_xlabel('Actual Values')
        axes[i, 0].set_ylabel('Predicted Values')
        axes[i, 0].set_title(f'Predictions vs. Actual for {model_names[i]}')
        
        stats.probplot(residuals, plot=axes[i, 1])
        axes[i, 1].set_title(f'QQ-Plot of Residuals for {model_names[i]}')

    plt.tight_layout()
    plot_path = f"results/{model_names[i]}_predictions_and_qqplot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

    summary_df = pd.DataFrame({
        'Model': model_names,
        'Best CV Score (Neg MSE)': best_scores,
        'MSE on Test Set': mse_scores,
        'R² on Test Set': r2_scores,
        'Best Parameters': best_params_list
    })

    os.makedirs("models_assignement_1", exist_ok=True)
    summary_path = "models_project/summary_regression_Overall.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, mse_scores, color='skyblue')
    plt.title('Comparison of Models based on MSE')
    plt.xlabel('Models')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    mse_bar_path = "results/comparison_mse.png"
    plt.savefig(mse_bar_path)
    print(f"Bar chart saved to {mse_bar_path}")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, r2_scores, color='lightgreen')
    plt.title('Comparison of Models based on R² Score')
    plt.ylim((0.98, 1.0))
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    r2_bar_path = "results/comparison_r2.png"
    plt.savefig(r2_bar_path)
    print(f"Bar chart saved to {r2_bar_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preprocessing and model pipeline.")
    parser.add_argument('--data_dir', type=str, default="data/Simusafe_Dataset", help="Path to the dataset directory.")
    parser.add_argument('--output_dir', type=str, default="preprocessed_data", help="Path to save preprocessed data.")
    parser.add_argument('--target_column', type=str, default="risk_evaluation", help="Target column for the pipeline.")
    parser.add_argument('--task_type', type=str, choices=["regression", "classification"], default="regression", help="Task type: 'regression' or 'classification'.")
    parser.add_argument('--subset_frac', type=float, default=1.0, help="Fraction of the data to use for training.")
    parser.add_argument('--random_state', type=int, default=42, help="Random state for reproducibility.")
    
    args = parser.parse_args()
    main(args)
