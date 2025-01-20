# utils.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import os 
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats

from imblearn.over_sampling import SMOTE

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
from sklearn.base import BaseEstimator, RegressorMixin, clone

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

class MixtureOfExperts(BaseEstimator, RegressorMixin):
    def __init__(self, experts=None):
        self.experts = experts if experts is not None else []
    
    def fit(self, X, y):
        self.experts_ = [clone(expert) for expert in self.experts]
        for expert in self.experts_:
            expert.fit(X, y)
        return self

    # def predict(self, X):
    #     predictions = np.column_stack([expert.predict(X) for expert in self.experts_])
    #     return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
        
    # def predict(self, X):
    #     predictions = np.column_stack([expert.predict(X) for expert in self.experts_])
    #     return predictions.mean(axis=1)

    def predict(self, X):
        predictions = np.column_stack([expert.predict(X) for expert in self.experts_])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
    
    def predict_proba(self, X):
        probas = np.column_stack([expert.predict_proba(X) for expert in self.experts_])
        return np.mean(probas, axis=1)
    


class WeightedMixtureOfExperts(BaseEstimator, RegressorMixin):
    def __init__(self, expert1=None, expert2=None):
        self.expert1 = expert1
        self.expert2 = expert2
        self.alpha_ = 0.5 
    
    def fit(self, X, y):
        self.expert1_ = clone(self.expert1)
        self.expert2_ = clone(self.expert2)
        
        self.expert1_.fit(X, y)
        self.expert2_.fit(X, y)
        
        y1 = self.expert1_.predict(X)
        y2 = self.expert2_.predict(X)
        
        p = y1 - y2
        numerator = np.sum(p * (y2 - y))
        denominator = np.sum(p**2)
        
        if denominator == 0:
            self.alpha_ = 0.5
        else:
            alpha_opt = - numerator / denominator
            self.alpha_ = np.clip(alpha_opt, 0.0, 1.0)
        
        return self
    
    def predict(self, X):
        y1 = self.expert1_.predict(X)
        y2 = self.expert2_.predict(X)
        return self.alpha_ * y1 + (1 - self.alpha_) * y2


class DynamicMixtureOfExperts(BaseEstimator, RegressorMixin):
    """
    A dynamic approach to building an ensemble of experts
    by iteratively adding or removing experts to minimize MSE.
    Uses a simple average of the selected experts' predictions.
    """

    def __init__(self, experts=None, max_iter=5, do_backward=True):
        """
        :param experts: List of scikit-learn regressor objects (unfitted).
        :param max_iter: Maximum number of forward-backward passes.
        :param do_backward: Whether to do backward elimination after forward selection.
        """
        self.experts = experts if experts is not None else []
        self.max_iter = max_iter
        self.do_backward = do_backward

    def fit(self, X, y):
        self.fitted_experts_ = [clone(exp).fit(X, y) for exp in self.experts]

        expert_preds = []
        for model in self.fitted_experts_:
            expert_preds.append(model.predict(X))
        self.expert_preds_ = np.column_stack(expert_preds)

        
        best_mse = float("inf")
        best_idx = 0
        n_experts = len(self.fitted_experts_)
        for i in range(n_experts):
            mse_i = mean_squared_error(y, self.expert_preds_[:, i])
            if mse_i < best_mse:
                best_mse = mse_i
                best_idx = i
        self.selected_ = {best_idx}  

        current_mse = best_mse
        for it in range(self.max_iter):
            improved = False

            # ========================
            # FORWARD SELECTION
            # ========================
            best_add_mse = current_mse
            best_add_idx = None
            for i in range(n_experts):
                if i not in self.selected_:
                    trial_subset = list(self.selected_) + [i]
                    y_ens = self._ensemble_predictions(self.expert_preds_, trial_subset)
                    mse_trial = mean_squared_error(y, y_ens)
                    if mse_trial < best_add_mse:
                        best_add_mse = mse_trial
                        best_add_idx = i

            if best_add_idx is not None and best_add_mse < current_mse:
                self.selected_.add(best_add_idx)
                current_mse = best_add_mse
                improved = True

            # ========================
            # BACKWARD ELIMINATION
            # ========================
            if self.do_backward and len(self.selected_) > 1:
                best_remove_mse = current_mse
                best_remove_idx = None
                for i in list(self.selected_):
                    trial_subset = list(self.selected_ - {i})
                    y_ens = self._ensemble_predictions(self.expert_preds_, trial_subset)
                    mse_trial = mean_squared_error(y, y_ens)
                    if mse_trial < best_remove_mse:
                        best_remove_mse = mse_trial
                        best_remove_idx = i
                if best_remove_idx is not None and best_remove_mse < current_mse:
                    self.selected_.remove(best_remove_idx)
                    current_mse = best_remove_mse
                    improved = True

            if not improved:
                break

        self.selected_ = sorted(self.selected_)
        self.current_mse_ = current_mse
        return self

    def predict(self, X):
        if not hasattr(self, "selected_"):
            raise RuntimeError("Model not fitted yet!")
        preds = [self.fitted_experts_[i].predict(X) for i in self.selected_]
        return np.mean(preds, axis=0)

    def _ensemble_predictions(self, expert_preds, subset):
        subset_preds = expert_preds[:, subset]  
        return np.mean(subset_preds, axis=1)


def model_pipeline(df, 
                   target, 
                   test_dataset=None,
                   task_type='regression', 
                   subset_frac=1.0, 
                   random_state=42):
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
                    # GridSearch over MixtureOfExperts is not directly supported.
                }
            ),
            (
                MLPRegressor(max_iter=500, random_state=random_state),
                {
                    'hidden_layer_sizes': [(64,), (128,), (64, 32)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'sgd'],
                    'alpha': [0.0001, 0.001],
                    'learning_rate': ['constant', 'adaptive']
                }
            )
        ]
        scoring_metric = 'neg_mean_squared_error'
        
    elif task_type == 'classification':
        models_list = [
            (
                LGBMClassifier(verbose=-1, random_state=random_state),
                {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7]
                }
            ),
            (
                SVC(),
                {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'probability': [True]
                }
            ),
            (
                RandomForestClassifier(random_state=random_state),
                {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            ),
            (
                LogisticRegression(random_state=random_state),
                {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [100, 200, 500]
                }
            ),
            (
                KNeighborsClassifier(),
                {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            ),
            (
                XGBClassifier(random_state=random_state, use_label_encoder=False),
                {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7]
                }
            ),
            (
                MixtureOfExperts(experts=[
                    LGBMClassifier(random_state=random_state),
                    RandomForestClassifier(random_state=random_state),
                    XGBClassifier(random_state=random_state, use_label_encoder=False)
                ]),
                {
                    # GridSearch over MixtureOfExperts is not directly supported.
                }
            ),
            (
                MLPClassifier(max_iter=500, random_state=random_state),
                {
                    'hidden_layer_sizes': [(64,), (128,), (64, 32)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'sgd'],
                    'alpha': [0.0001, 0.001],
                    'learning_rate': ['constant', 'adaptive']
                }
            ),
        ]
        scoring_metric = 'accuracy'

    else:
        raise ValueError("task_type must be either 'regression' or 'classification'")

    if subset_frac:
        df_sampled = df.sample(frac=subset_frac, random_state=random_state)
        X = df_sampled.drop(columns=target)
        y = df_sampled[target]
    else:
        X = df.drop(columns=target)
        y = df[target]


    if task_type == 'classification':
        rare_class_value = 2  
        rare_class_rows = X[y == rare_class_value]
        
        rare_class_noisy = rare_class_rows.copy()
        noise_scale = 0.1

        for _ in range(20): 
            rare_class_numeric = rare_class_rows.select_dtypes(include=[np.number])
            noisy_numeric = rare_class_numeric + np.random.normal(0, noise_scale, rare_class_numeric.shape)

            noisy_rows = rare_class_rows.copy()
            noisy_rows[rare_class_numeric.columns] = noisy_numeric

            rare_class_noisy = pd.concat([rare_class_noisy, noisy_rows], ignore_index=True)

        X = pd.concat([X, rare_class_noisy], ignore_index=True)
        y = pd.concat([y, pd.Series([rare_class_value] * len(rare_class_noisy))], ignore_index=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_numeric = X_train[numeric_cols]

        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_numeric, y_train)

        X_train_resampled = pd.DataFrame(X_train_resampled, columns=numeric_cols)
        X_train_resampled = pd.concat([X_train.drop(columns=numeric_cols).reset_index(drop=True), 
                                    X_train_resampled.reset_index(drop=True)], axis=1)

    else:
        if test_dataset is not None:
            print("Using testing dataset as X_test/y_test")
            X_train = X
            y_train = y
            X_test = test_dataset.drop(columns=target)
            y_test = test_dataset[target]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
        X_train_resampled, y_train_resampled = X_train, y_train
    for class_label, count in y.value_counts().items():
        print(f"Class {class_label}: {count} instances")

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
                n_jobs=16
            )
            
            grid_search.fit(X_train_resampled, y_train_resampled)

            best_score = -grid_search.best_score_ if scoring_metric == 'neg_mean_squared_error' else grid_search.best_score_

            print(f"Best CV score for {pipeline.named_steps['model'].__class__.__name__}: {best_score:.4f}")
            print(f"Best parameters: {grid_search.best_params_}\n")

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            result_dict = {
                'model': pipeline.named_steps['model'].__class__.__name__,
                'best_score': best_score,
                'best_params': grid_search.best_params_,
                'X_train': X_train_resampled,
                'y_train': y_train_resampled,
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


def plot_model_comparison(results, metrics, output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        model_names = [result['Model'] for result in results]
    except KeyError:
        raise ValueError("Missing 'Model' key in results. Ensure the input data has 'Model' keys.")

    for metric in metrics:
        metric_values = [result.get(metric, 0) for result in results]
        min_metric_value = min(metric_values)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, metric_values, color='skyblue')
        plt.title(f'Comparison of Models based on {metric}')
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.ylim((min_metric_value - 0.03, 1.0))
        plt.xticks(rotation=45)
        
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
                     f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"comparison_{metric.lower().replace(' ', '_')}.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"{metric} comparison plot saved to {plot_path}")
        plt.close()


