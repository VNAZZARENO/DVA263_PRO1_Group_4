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
python main.py --data_dir data/ --output_dir preprocessed_data/ --target_column risk_evaluation --task_type regression --subset_frac 1.0 --random_state 42
```

### **Arguments:**
- `--data_dir`: Path to the directory containing the dataset files (default: `data/`).
- `--output_dir`: Path to save the preprocessed data (default: `preprocessed_data/`).
- `--target_column`: The column to predict (`risk_evaluation` in this case).
- `--task_type`: Type of task (`regression` or `classification`).
- `--subset_frac`: Fraction of the dataset to use (default: `1.0` for all data).
- `--random_state`: Random seed for reproducibility (default: `42`).

---

Here’s the **Results** section updated to display images using direct links to the repository:

---

## **Results**

### **Model Comparison**

The table below summarizes the performance of the models tested for the regression task, ranked by their performance on the test set:

| **Model**           | **Best CV Score (Neg MSE)** | **MSE on Test Set** | **R² on Test Set** | **Best Parameters**                                                                                     |
|----------------------|-----------------------------|----------------------|--------------------|---------------------------------------------------------------------------------------------------------|
| **MLPRegressor**     | 0.000180                   | 0.000086            | 0.9983            | `{'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (64,), 'learning_rate': 'constant', 'solver': 'adam'}` |
| **MixtureOfExperts** | 0.000191                   | 0.000179            | 0.9964            | `{}`                                                                                                   |
| **LGBMRegressor**    | 0.000194                   | 0.000252            | 0.9949            | `{'learning_rate': 0.1, 'n_estimators': 200}`                                                          |


1. **MLPRegressor**:
   - Achieved the best performance with an **MSE of 0.000086** and **R² of 0.9983**.
   - Best parameters include a single hidden layer with 64 units, **ReLU activation**, and the **Adam optimizer** with a constant learning rate.

2. **MixtureOfExperts**:
   - Performed well with an **MSE of 0.000179** and **R² of 0.9964**.
   - The model dynamically combined predictions from multiple experts, showcasing robustness but slightly lagging behind MLP.

3. **LGBMRegressor**:
   - Delivered solid performance with an **MSE of 0.000252** and **R² of 0.9949**.
   - Optimized parameters included a learning rate of 0.1 and 200 estimators, but it was outperformed by the MLP and Mixture of Experts.

---

### **Visualizations**

1. **MSE Comparison**:  
   ![MSE Comparison](https://github.com/VNAZZARENO/DVA263_PRO1_Group_4/blob/main/results/comparison_mse.png?raw=true)

2. **R² Score Comparison**:  
   ![R² Comparison](https://github.com/VNAZZARENO/DVA263_PRO1_Group_4/blob/main/results/comparison_r2.png?raw=true)

3. **Predictions and Residuals**:
   ![All models](https://github.com/VNAZZARENO/DVA263_PRO1_Group_4/blob/main/results/MLPRegressor_predictions_and_qqplot.png?raw=true)
   
---

### **Conclusion**

- The **MLPRegressor** showed much better performance overall in terms of both MSE and R² score, making it the most effective model for the `risk_evaluation` prediction.
- The **Mixture of Experts** approach provided better results than simple LGBM and can still be used in scenarios requiring robustness and results explanations.
- The **LightGBM Regressor** remains a strong baseline but was outperformed by a combination of LGBM + Linear Regression.

---

## **Contact**
For any questions, reach out to:  
- Vincent Nazzareno: vincent.nazzareno@gmail.com  
- Nicola Baldoni: nicola.baldoni.01@gmail.com  
