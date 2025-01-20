# main.py
import os
import logging
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats

from data.utils import model_pipeline, plot_model_comparison
from data.preprocessing import preprocess_data, create_features, apply_pca

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score, classification_report

logging.basicConfig(level=logging.INFO)


import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

def main(args):
    import os
    import pandas as pd
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import scipy.stats as stats

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
    df_merged = create_features(track_data)

    if args.drop_nan_columns:
        drop_col_override = ["hr", "hrv_lf", "hrv_hf", 'hrv_lfhf_ratio', 'EBRmean', 'BDmean']
        print(f"Dropping columns: {drop_col_override}")


        row_index_to_drop = [
            index for index, row in df_merged.iterrows()
            if row.isna().sum() > len(df_merged.columns) * 0.75
        ]
        df_merged.drop(index=row_index_to_drop, inplace=True)
        df_merged.drop(columns=drop_col_override, axis=1, inplace=True)
        df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)

    if args.use_pca == True:
        print("USING PCA:")
        target_col_data = df_merged[args.target_column]
        df_features = df_merged.drop(columns=[args.target_column])

        transformed_data, pca_model = apply_pca(df_features, n_components=5)
        print("Explained Variance Ratio:", pca_model.explained_variance_ratio_)
        print("PCA Transformed Data Shape:", transformed_data.shape)

        df_merged = pd.DataFrame(transformed_data)
        df_merged[args.target_column] = target_col_data.reset_index(drop=True)

        if args.use_test_dataset:
            print("USING TESTING DATASET:")
            df_test = create_features(simulation_data)
            target_test_col = df_test[args.target_column]
            df_test_features = df_test.drop(columns=[args.target_column])
            transformed_test_data, _ = apply_pca(df_test_features, n_components=5)
            df_test = pd.DataFrame(transformed_test_data)
            df_test[args.target_column] = target_test_col.reset_index(drop=True)

    results = model_pipeline(
        df=df_merged,
        target=args.target_column,
        test_dataset=(df_test if args.use_test_dataset else None),
        task_type=args.task_type,
        subset_frac=args.subset_frac,
        random_state=args.random_state
    )

    summary_filename = f"models_project/summary_overall_{args.task_type}.csv"

    if args.task_type == "classification":
        detailed_reports = []

        for result in results:
            y_test = result['y_test']
            y_pred = result['y_pred']

            report = classification_report(y_test, y_pred, output_dict=True)
            detailed_report_str = classification_report(y_test, y_pred)
            detailed_reports.append({
                "Model": result['model'],
                "Precision": report['weighted avg']['precision'],
                "Recall": report['weighted avg']['recall'],
                "F1-Score": report['weighted avg']['f1-score'],
                "Support": report['weighted avg']['support'],
                "Classification Report": detailed_report_str
            })

            report_path = f"results/{result['model']}_classification_report_{args.task_type}.txt"
            with open(report_path, "w") as f:
                f.write(detailed_report_str)
            print(f"Classification report saved to {report_path}")

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='viridis')
            plt.title(f"Confusion Matrix for {result['model']}")
            plot_path = os.path.join("results/plots/", f"{result['model']}_confusion_matrix_{args.task_type}.png")
            plt.savefig(plot_path)
            print(f"Confusion matrix saved to {plot_path}")
            plt.show()

        classification_summaries = pd.DataFrame(detailed_reports)
        classification_summaries.to_csv(f"results/classification_reports_summary_{args.task_type}.csv", index=False)
        
    elif args.task_type == "regression":
        num_models = len(results)
        fig, axes = plt.subplots(num_models, 2, figsize=(12, 4 * num_models), squeeze=False)

        for i in range(num_models):
            y_test = results[i]['y_test']
            y_pred = results[i]['y_pred']
            residuals = y_test - y_pred

            axes[i, 0].scatter(y_test, y_pred, color='blue', alpha=0.5)
            axes[i, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            axes[i, 0].set_xlabel('Actual Values')
            axes[i, 0].set_ylabel('Predicted Values')
            axes[i, 0].set_title(f'Predictions vs. Actual for {results[i]["model"]}')

            stats.probplot(residuals, plot=axes[i, 1])
            axes[i, 1].set_title(f'QQ-Plot of Residuals for {results[i]["model"]}')

        plt.tight_layout()
        plot_path = f"results/{results[i]['model']}_predictions_and_qqplot.png"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.show()

    summary_df = pd.DataFrame({
        'Model': [result['model'] for result in results],
        'Best CV Score (Neg MSE)' if args.task_type == "regression" else 'Accuracy': [result['best_score'] for result in results],
        'MSE on Test Set' if args.task_type == "regression" else 'N/A': [result['mse'] if args.task_type == "regression" else None for result in results],
        'R² on Test Set' if args.task_type == "regression" else 'N/A': [result['r2'] if args.task_type == "regression" else None for result in results],
        'Best Parameters': [result['best_params'] for result in results]
    })

    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary saved to {summary_filename}")

    if args.task_type == "regression":
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df['Model'], summary_df['MSE on Test Set'], color='skyblue')
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
        plt.bar(summary_df['Model'], summary_df['R² on Test Set'], color='lightgreen')
        plt.title('Comparison of Models based on R² Score')
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        r2_bar_path = "results/comparison_r2.png"
        plt.savefig(r2_bar_path)
        print(f"Bar chart saved to {r2_bar_path}")
        plt.show()


    if args.task_type == "classification":
        classification_summaries = pd.DataFrame(detailed_reports)
        classification_summaries['Model'] = [result['model'] for result in results]
        classification_summaries.to_csv(f"results/classification_reports_summary_{args.task_type}.csv", index=False)

        metrics_to_compare = ["Precision", "Recall", "F1-Score", "Accuracy"]
        plot_model_comparison(classification_summaries.to_dict('records'), metrics_to_compare)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preprocessing and model pipeline.")
    parser.add_argument('--data_dir', type=str, default="data/Simusafe_Dataset", help="Path to the dataset directory.")
    parser.add_argument('--output_dir', type=str, default="preprocessed_data/", help="Path to save preprocessed data.")
    parser.add_argument('--target_column', type=str, default="risk_evaluation", help="Target column for the pipeline.")
    parser.add_argument('--task_type', type=str, choices=["regression", "classification"], default="regression", help="Task type: 'regression' or 'classification'.")
    parser.add_argument('--use_pca', type=bool, default=True, help="Use PCA: Use PCA for Feature Reduction")
    parser.add_argument('--use_test_dataset', type=bool, default=False, help="Use or not Simulation data as a testing dataset")
    parser.add_argument('--subset_frac', type=float, default=None, help="Fraction of the data to use for training (to be quicker).")
    parser.add_argument('--drop_nan_columns', type=bool, default=False, help="Drop the columns which are >75% NaN values.")
    parser.add_argument('--random_state', type=int, default=42, help="Random state for reproducibility.")
    
    args = parser.parse_args()
    main(args)


