# ML_Coursework1

**Author:** Junhao Zhou (k23172173)  
**Date:** February 16th, 2026

## Project Overview
The objective of this coursework is to develop a robust regression model to predict the target variable `outcome` based on a dataset of diamond characteristics. The project involves extensive exploratory data analysis (EDA), rigorous data cleaning, and the implementation of a **Random Forest Regressor** to minimize generalization error on a held-out test set.

## Repository Structure
* `data_clean_rfmodel.py`: The main script containing the data cleaning pipeline, feature engineering, model training, and prediction generation.
* `visualization.py`: Script used to generate EDA plots and "Before vs. After" cleaning comparisons.
* `CW1_train.csv`: Training dataset.
* `CW1_test.csv`: Test dataset (unlabeled).
* `CW1_submission_23172173.csv`: Final prediction output.
* `requirements.txt`: List of Python dependencies.
* `ML_Coursework1_Report.pdf`: Detailed academic report of the methodology and results.

## Methodology

### 1. Data Cleaning Pipeline
To ensure model robustness, a three-stage cleaning strategy was implemented to handle noise and outliers:
* **Logical Constraints:** Removed instances where `carat <= 0` or `carat > 5`.
* **Physical Constraints:** Removed rows with zero dimensions (`x`, `y`, or `z` = 0), as diamonds cannot have zero volume.
* **Statistical Outliers (IQR Method):** Applied to `depth` and `table` features. Data points falling outside $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$ were identified as noise and removed.

### 2. Feature Engineering
* **Ordinal Encoding:** Categorical variables (`cut`, `color`, `clarity`) were mapped to numerical values preserving their inherent hierarchy (e.g., Ideal > Premium > ...).

### 3. Model Selection
**Random Forest Regressor** was selected as the final model due to its ability to:
* Handle non-linear relationships between anonymized features and the outcome.
* Process mixed data types (numerical and categorical) effectively.
* Provide an unbiased estimate of generalization error via Out-of-Bag (OOB) scoring.

## Model Performance
The model performance was evaluated using the Coefficient of Determination ($R^2$). The **Out-of-Bag (OOB) Score** is used as the primary metric for estimating performance on unseen data.

| Metric | Score ($R^2$) | Description |
| :--- | :--- | :--- |
| **Random Forest OOB Score** | **0.45034** | **Primary validation metric (Estimated Test $R^2$)** |
| Random Forest Training $R^2$ | 0.92486 | Indicates strong fit to training data |

## Visualizations
Key visualizations generated during the analysis include:
* **Boxplots:** Comparing feature distributions before and after IQR cleaning.
* **Scatter Plots:** Bivariate analysis of Depth vs. Table showing outlier removal.
* **Feature Importance:** Highlighting the most predictive variables in the Random Forest model.
