# **Risk Evaluation Prediction using Machine Learning**

This project focuses on predicting the `risk_evaluation` metric from the car dataset. The goal is to build an end-to-end machine learning pipeline, including preprocessing, feature engineering, model training, and evaluation, while comparing multiple regression models.

---

## **Project Workflow**

### 1. **Preprocessing and Feature Engineering**
- Handles missing values and irrelevant columns.
- Creates new features:
  - **Total Violations**, **Number of Emotions**, **Distraction Level**, **Gaze Level**, **Speed Ratio**, etc.
  
### 2. **Model Development**
- Implements and compares models such as:
  - **LightGBM Regressor**
  - **Mixture of Experts**
  - **Dynamic Mixture of Experts**
  - **Weighted Mixture of Experts**

### 3. **Evaluation**
- Measures performance using:
  - **Mean Squared Error (MSE)**
  - **R² Score**
- Visualizes predictions vs actual values and residuals using QQ plots.

---

## **Key Files**

- **`main.py`**: The main script to preprocess data, train models, and generate evaluation results.
- **`preprocessing.py`**: Contains preprocessing and feature engineering pipelines.
- **`utils.py`**: Implements models like Mixture of Experts and evaluation utilities.
- **`results/`**: Stores visualizations and model evaluation summaries (e.g., MSE bar plots, QQ plots).
- **`models_project/`**: Stores the best trained model for further usage.

---

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone DVA263_PRO1_Group_4.git
cd DVA263_PRO1_Group_4
```

### **2. Install Requirements**
Ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `lightgbm`
- `tqdm`

Install them using:
```bash
pip install -r requirements.txt
```

### **3. Prepare the Dataset**
- Place the dataset files (`About_the_dataset.xlsx`, `Feature_Simulation.xlsx`, `Feature_Track.xlsx`) in the `data/` directory.

---

## **Run the Project**

### **Execute Main Script**
Run the entire pipeline (data preprocessing, model training, evaluation, and visualization) with:
```bash
python main.py \
    --data_dir data/ \
    --output_dir preprocessed_data/ \
    --target_column risk_evaluation \
    --task_type regression \
    --subset_frac 1.0 \
    --random_state 42
```

### **Arguments:**
- `--data_dir`: Path to the directory containing the dataset files (default: `data/`).
- `--output_dir`: Path to save the preprocessed data (default: `preprocessed_data/`).
- `--target_column`: The column to predict (`risk_evaluation` in this case).
- `--task_type`: Type of task (`regression` or `classification`).
- `--subset_frac`: Fraction of the dataset to use (default: `1.0` for all data).
- `--random_state`: Random seed for reproducibility (default: `42`).

---

## **Results**

### **Model Comparisons**
- **MSE Comparison**: See `results/comparison_mse.png`.
- **R² Comparison**: See `results/comparison_r2.png`.

### **Example Plots**
- LightGBM Regressor: `results/LGBMRegressor_predictions_and_qqplot.png`
- Mixture of Experts: `results/MixtureOfExperts_predictions_and_qqplot.png`
- Weighted Mixture of Experts: `results/WeightedMixtureOfExperts_predictions_and_qqplot.png`

---

## **Contact**
For any questions, reach out to:  
- Vincent Nazzareno: vincent.nazzareno@gmail.com  
- Nicola Baldoni: nicola.baldoni.01@gmail.com  
