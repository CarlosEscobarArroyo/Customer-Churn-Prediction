# 🔁 Customer Churn Prediction

**Project status:** In development and refinement

This repository contains the code and tools used to develop a **customer churn prediction model** for a **real company in the direct sales sector**. The goal is to anticipate which sales representatives (vendedoras) may stop collaborating with the company, enabling the business to implement retention and loyalty strategies.

Churn is a key metric in any business. Even small increases in churn can significantly impact revenue and raise the cost of acquiring new customers. Reducing churn improves **Customer Acquisition Cost (CAC)** efficiency, as retaining existing users is typically less expensive than acquiring new ones.

---

## ⚠️ Data Disclaimer

Because this project is based on **sensitive real-world data**, the **original datasets are not included**.

Instead, we provide anonymized example datasets to illustrate the structure and flow of the pipeline:

- `data/CHURN_FEATURES_EXAMPLE.csv`: Example of the final feature set used for training.
- `data/VALIDATION_FEATURES_EXAMPLE.csv`: Example input for external validation.
- `reports/resultados_validacion_EXAMPLE.xlsx`: Example of output results from model validation.

These files contain **mocked or anonymized data** and can be used to explore the notebooks and pipelines in this repository.

---

## 📓 Notebook Descriptions

- **EDA_General.ipynb**: Descriptive statistics, missing values, class balance, correlation matrix, outlier detection using IQR.
- **ModelsExperimentation.ipynb**: Preprocessing (encoding, scaling), class balancing (SMOTE + Tomek Links), model comparison (Logistic Regression, Decision Trees, Random Forest, SVM), and hyperparameter tuning with GridSearchCV.
- **SVC_Pipeline.ipynb**: End-to-end pipeline using `ColumnTransformer`, one-hot encoding, scaling, and SVM classification.
- **Statistical_Tests.ipynb**: Performs statistical hypothesis testing (Levene and Welch’s t-test) to compare numerical features between churned and non-churned users.

---

## 🧰 Helper Functions (`src/support_functions.py`)

- `plot_distribution()`: Overlapped histograms by class.
- `plot_correlation_matrix()`: Correlation heatmap.
- `plot_outliers()`: Boxplots to detect outliers.
- `plot_class_balance()`: Bar and pie charts for class balance.
- `plot_confusion_matrices()` and `plot_roc_curves()`: Visual comparison of model performance.
- `detect_outliers_iqr()` and `erase_outliers_iqr()`: Outlier detection/removal using IQR method.

---

## 📋 Dataset Features (example schema)

| Type       | Name                          | Brief Description                                          |
|------------|-------------------------------|------------------------------------------------------------|
| Numerical  | `ANTIGUEDAD_MESES`            | Seller's tenure (in months) at the company                |
| Numerical  | `MONTO_ACTUAL`                | Current sales or order amount                             |
| Numerical  | `MONTO_ANTERIOR`              | Previous period's order amount                            |
| Numerical  | `NUM_CAMPANAS_HISTORICAS`     | Total number of campaigns participated in                 |
| Numerical  | `CAMPANAS_CONSECUTIVAS_PREVIAS` | Number of recent consecutive campaigns participated in |
| Categorical| `DEPARTAMENTO`                | Geographic region or department (one-hot encoded)         |
| Binary     | `SEXO`                        | Seller's gender (F/M mapped to 0/1)                        |
| Binary     | `TIPO_VENDEDOR`               | Role within sales network (Asesora/Líder)                 |
| Binary     | `ES_NUEVA`                    | Indicator if the seller is new                            |
| Binary     | `ES_INICIO_ANIO`              | Indicates if sale occurs at the start of the year         |
| Binary     | `ES_NAVIDAD`                  | Indicates if sale occurs during the Christmas campaign    |
| Target     | `TARGET_CHURN`                | Binary target: 1 = churned, 0 = retained                   |

> 🧽 Note: Fields such as `ID_VENDEDOR`, `ANIO`, or `EDAD_VENDEDORA` were excluded from the example files to prevent bias or due to quality issues.
