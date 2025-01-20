# **Risk Evaluation Prediction using Machine Learning**

## **Project Overview**
This project predicts the `risk_evaluation` metric from a car dataset using machine learning. It involves data preprocessing, feature engineering, model training, and evaluation with classification techniques.

---

## **Project Workflow**

### 1. **Preprocessing and Feature Engineering**
- Handles missing values and removes irrelevant columns.
- New features created include:
  - **Total Violations**
  - **Number of Emotions**
  - **Distraction Level**
  - **Gaze Level**
  - **Speed Ratio**
- PCA is optionally applied for dimensionality reduction.

### 2. **Model Development**
- Models tested for classification:
  - **LightGBM Classifier**
    - Hyperparameters: Learning rate, number of estimators, max depth.
  - **Support Vector Classifier (SVC)**
    - Hyperparameters: Regularization (`C`), kernel type, probability.
  - **Random Forest Classifier**
    - Hyperparameters: Number of estimators, max depth, minimum samples split.
  - **Logistic Regression**
    - Hyperparameters: Regularization (`C`), solver, max iterations.
  - **K-Nearest Neighbors (KNN)**
    - Hyperparameters: Number of neighbors, weights, metric.
  - **XGBoost Classifier**
    - Hyperparameters: Learning rate, number of estimators, max depth.
  - **Mixture of Experts**
    - Combines predictions from LightGBM, Random Forest, and XGBoost.
  - **MLP Classifier**
    - Hyperparameters: Hidden layer sizes, activation function, solver, learning rate.

---

## **Directory Structure**

- **`main.py`**: Orchestrates the full pipeline (preprocessing, training, evaluation, visualization).
- **`preprocessing.py`**: Handles preprocessing, feature engineering, and PCA application.
- **`utils.py`**: Implements custom models, utility functions, and evaluation metrics.
- **`data/`**: Contains input datasets:
  - `About_the_dataset.xlsx`
  - `Feature_Simulation.xlsx`
  - `Feature_Track.xlsx`
- **`results/`**: Stores evaluation summaries and visualizations.
- **`models_project/`**: Saves trained models for future use.

---

## **Getting Started**

### 1. Clone the Repository
```bash
git clone DVA263_PRO1_Group_4.git
cd DVA263_PRO1_Group_4
```

### 2. Install Requirements
Install necessary Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Place the dataset files in the `data/` directory:
- `About_the_dataset.xlsx`
- `Feature_Simulation.xlsx`
- `Feature_Track.xlsx`

### 4. Run the Project
Execute the pipeline with:
```bash
python main.py --target_column risk_evaluation --task_type regression --use_pca True --drop_nan_columns True --random_state 42
```

#### Key Arguments:
- `--data_dir`: Path to dataset directory (default: `data/`).
- `--output_dir`: Directory for preprocessed data (default: `preprocessed_data/`).
- `--target_column`: Column to predict (default: `risk_evaluation`).
- `--task_type`: Task type: `regression` or `classification`.
- `--use_pca`: Whether to apply PCA (default: `False`).
- `--drop_nan_columns`: Drop columns with >75% NaN values (default: `False`).
- `--random_state`: Seed for reproducibility (default: `42`).

---

## **Contact**
For questions, contact:
- **Vincent Nazzareno**: vincent.nazzareno@gmail.com
- **Nicola Baldoni**: nicola.baldoni.01@gmail.com
