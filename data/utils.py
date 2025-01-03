import numpy as np
from tqdm import tqdm
import os 
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats


from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV



class MixtureOfExperts(BaseEstimator, RegressorMixin):
    def __init__(self, experts=None):
        self.experts = experts if experts is not None else []
    
    def fit(self, X, y):
        self.experts_ = [clone(expert) for expert in self.experts]
        for expert in self.experts_:
            expert.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = np.column_stack([expert.predict(X) for expert in self.experts_])
        return predictions.mean(axis=1)

def model_pipeline(df, target, task_type='regression', subset_frac=1.0, random_state=42):
    """
    Simplified pipeline focusing only on LightGBM and a MixtureOfExperts for regression or classification.
    """
    
    # Identify categorical and numerical columns
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop(target)
    except:
        categorical_cols = []

    try:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = numerical_cols.drop(target)
    except:
        numerical_cols = []

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )


    if task_type == 'regression':
        models_list = [
            (
                LGBMRegressor(random_state=random_state),
                {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200]
                }
            ),
            (
                MixtureOfExperts(experts=[
                    LGBMRegressor(random_state=random_state),
                    LinearRegression()
                ]),
                {
                    # 'experts__0__learning_rate': [0.01, 0.1], # Not directly supported by default GridSearch 
                }
            )
        ]
        scoring_metric = 'neg_mean_squared_error'
        
    elif task_type == 'classification':
        # Example classification version with LightGBM + a simple MoE (LightGBM + SVC)
        models_list = [
            (
                LGBMClassifier(random_state=random_state),
                {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200]
                }
            ),
            (
                MixtureOfExperts(experts=[
                    LGBMClassifier(random_state=random_state),
                    SVC(probability=True)
                ]),
                {
                }
            )
        ]
        scoring_metric = 'accuracy'
    else:
        raise ValueError("task_type must be either 'regression' or 'classification'")

    df_sampled = df.sample(frac=subset_frac, random_state=random_state)

    X = df_sampled.drop(columns=target)
    y = df_sampled[target]

    if task_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

    pipelines = [
        Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]) for model, _ in models_list
    ]

    param_grids = [
        {
            f'model__{param}': values 
            for param, values in params.items()
        }
        for _, params in models_list
    ]

    results = []

    for i, (pipeline, param_grid) in tqdm(enumerate(zip(pipelines, param_grids)), total=len(pipelines)):

        try:
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=5, 
                scoring=scoring_metric, 
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)

            best_score = -grid_search.best_score_ if scoring_metric == 'neg_mean_squared_error' else grid_search.best_score_

            print(f"Best CV score for {pipeline.named_steps['model'].__class__.__name__}: {best_score:.4f}")
            print(f"Best parameters: {grid_search.best_params_}\n")

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            result_dict = {
                'model': pipeline.named_steps['model'].__class__.__name__,
                'best_score': best_score,
                'best_params': grid_search.best_params_,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'best_model': best_model
            }

            if task_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"Test Set Metrics for {pipeline.named_steps['model'].__class__.__name__}:")
                print(f"MSE: {mse:.4f}, R2: {r2:.4f}\n")
                result_dict['mse_test'] = mse
                result_dict['r2_test'] = r2
            else:
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Test Set Metrics for {pipeline.named_steps['model'].__class__.__name__}:")
                print(f"Accuracy: {accuracy:.4f}\n")
                print(classification_report(y_test, y_pred))
                result_dict['accuracy_test'] = accuracy

            print("-"*30)
            results.append(result_dict)

        except Exception as e:
            print(f"An error occurred with model {i+1}: {e}")

    if len(results) == 0:
        raise RuntimeError("No valid results were obtained from the models.")
    
    if scoring_metric == 'neg_mean_squared_error':
        best_model_result = min(results, key=lambda x: x['best_score'])
    else:
        best_model_result = max(results, key=lambda x: x['best_score'])

    print(f"Best model overall: {best_model_result['model']}, score: {best_model_result['best_score']:.4f}")
    print(f"Best parameters: {best_model_result['best_params']}\n")

    os.makedirs("models_project", exist_ok=True)
    best_model_name = f"models_project/best_model_{task_type}_{target}.joblib"
    joblib.dump(best_model_result['best_model'], best_model_name)
    print(f"Best model saved as {best_model_name}")

    return results




def plot_results(results):
    for _ , res in enumerate(results):
        model_name = res['model']
        y_test = res['y_test']
        y_pred = res['y_pred']
        residuals = y_test - y_pred

        plt.figure(figsize=(12,5))
    
        plt.subplot(1,2,1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.title(f"{model_name} - Predictions vs. Actual")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
    
        plt.subplot(1,2,2)
        stats.probplot(residuals, plot=plt)
        plt.title(f"{model_name} - Residuals QQ-Plot")
    
        plt.tight_layout()
        plt.show()