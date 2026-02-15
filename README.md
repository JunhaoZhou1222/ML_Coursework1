# ML_Coursework1

**Author:** Junhao Zhou (k23172173)  
**Date:** February 16th, 2026

## Project Overview
The objective of this coursework is to develop a robust regression model to predict the target variable `outcome` based on a dataset of diamond characteristics. The project involves extensive exploratory data analysis (EDA), rigorous data cleaning, and the implementation of a **XGboost** to minimize generalization error on a held-out test set.

## Repository Structure
* `data_clean_best_model.py`: The main script containing the data cleaning pipeline, feature engineering, model training, and prediction generation.
* `visualization.py`: Script used to generate EDA plots and "Before vs. After" cleaning comparisons.
* `CW1_train.csv`: Training dataset.
* `CW1_test.csv`: Test dataset (unlabeled).
* `CW1_submission_23172173.csv`: Final prediction output.
* `requirements.txt`: List of Python dependencies.

## Methodology

### 1. Data Cleaning Pipeline
To ensure model robustness, a three-stage cleaning strategy was implemented to handle noise and outliers:
* **Logical Constraints:** Removed instances where `carat <= 0` or `carat > 5`.
* **Physical Constraints:** Removed rows with zero dimensions (`x`, `y`, or `z` = 0), as diamonds cannot have zero volume.
* **Statistical Outliers (IQR Method):** Applied to `depth` and `table` features. Data points falling outside $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$ were identified as noise and removed.

### 2. Feature Engineering
* **Ordinal Encoding:** Categorical variables (`cut`, `color`, `clarity`) were mapped to numerical values preserving their inherent hierarchy (e.g., Ideal > Premium > ...).

### 3. Model Selection
**XGBoost** was selected as the final model due to its ability to:
* Superior Performance: It achieved the highest $R^2$ score on the validation set.
* Regularization: XGBoost utilizes L1/L2 regularization to prevent overfitting, offering an advantage over standard bagging methods like Random Forest.
* Gradient Boosting Framework: The sequential tree-building process allowed the model to correct residual errors more effectively than independent decision trees.

## Model Performance
The model performance was evaluated using the Coefficient of Determination ($R^2$) on a strictly held-out Validation Set (20% of training data).

| Metric | Score ($R^2$) | Description |
| :--- | :--- | :--- |
| **Linear Regression Score** | **0.32128** | **baseline model** |
| **XGboost Score** | **0.47015** | **Best performing model** |
| **Random Forest Score** | **0.45034** | **comparison model** |
| **Decision Tree Score** | **0.26777** | **Single Tree model** |
| **KNN Score** | **0.17860** | **Distance measure model** |
| **SVR Score** | **0.22279** | **Vector Regression model** |
| Training $R^2$ | 0.92486 | Indicates strong fit to training data |

## Visualizations
Key visualizations generated during the analysis include:
* **Boxplots:** Comparing feature distributions before and after IQR cleaning.
* **Scatter Plots:** Bivariate analysis of Depth vs. Table showing outlier removal.
* **Feature Importance:** Highlighting the most predictive variables in the XGboost model.
