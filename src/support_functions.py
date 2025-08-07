import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings


def plot_distribution(data, columns, target_col, figsize=(8, 4)):
    """
    Plots the distribution of numerical features grouped by target class using histograms.

    For each specified column, the function creates a histogram showing how the feature's 
    values are distributed across different classes of the target variable. This is useful
    for comparing class-wise distributions and detecting feature separation or skewness.

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data.
    columns : list of str
        List of numerical column names to plot.
    target_col : str
        The name of the target or grouping column (categorical or binary).
    figsize : tuple, optional
        Size of the entire figure (width, height). Default is (8, 4).

    Returns:
    -------
    None
        Displays a grid of overlaid histograms for each specified feature, grouped by class.
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for i, col in enumerate(columns):
        if i < len(axes):
            for target_val in data[target_col].unique():
                subset = data[data[target_col] == target_val]
                axes[i].hist(subset[col], alpha=0.7, label=f'Class {target_val}', bins=20)

            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].legend()

    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data, figsize=(8, 4)):
    """
    Plots the correlation matrix of numerical features in a DataFrame as a heatmap.

    The upper triangle of the matrix is masked to avoid redundancy. The heatmap uses
    a diverging color palette centered at 0, making it easier to interpret both positive 
    and negative correlations.

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing numerical features.
    figsize : tuple, optional
        Size of the figure (width, height). Default is (8, 4).

    Returns:
    -------
    pandas.DataFrame
        The correlation matrix computed from the input DataFrame.
    """
    corr_matrix = data.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})

    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return corr_matrix

def plot_outliers(data, columns, target_col, figsize=(8, 4)):
    """
    Plots boxplots of specified numerical columns grouped by a target column to visualize potential outliers.

    This function is useful for comparing the distribution of numerical features across target classes
    (e.g., churn vs no churn) and identifying the presence of outliers.

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data.
    columns : list of str
        List of numerical column names to plot.
    target_col : str
        The name of the target or grouping column (categorical or binary).
    figsize : tuple, optional
        Size of the entire figure (width, height). Default is (8, 4).

    Returns:
    -------
    None
        Displays grouped boxplots for each specified numerical column.
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for i, col in enumerate(columns):
        if i < len(axes):
            sns.boxplot(data=data, x=target_col, y=col, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')

    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_class_balance(y, title="Classes Distribution"):
    """
    Plots the class distribution of a binary or multiclass target variable using both bar and pie charts.

    This visualization is useful to assess class imbalance in classification tasks.
    The left subplot shows the count of each class, and the right subplot shows the percentage.

    Parameters:
    ----------
    y : array-like or pandas.Series
        The target variable containing class labels.
    title : str, optional
        Title prefix for the plots. Default is "Classes Distribution".

    Returns:
    -------
    None
        Displays the class distribution using bar and pie charts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    sns.set_style("whitegrid")

    # Bar chart
    value_counts = y.value_counts()
    ax1.bar(value_counts.index, value_counts.values, color='skyblue')
    ax1.set_title(f'{title} - Count')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Quantity')

    # Pie chart
    ax2.pie(value_counts.values, labels=[f'Class {i}' for i in value_counts.index],
            autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    ax2.set_title(f'{title} - Percentage')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(models_results, y_test, figsize=(15, 12)):
    """
    Plots confusion matrices for multiple classification models to compare their performance.

    Each subplot represents the confusion matrix of a different model, displayed as a heatmap.
    The function expects a dictionary where each entry contains a key 'predictions' 
    with the predicted class labels of the model.

    Parameters:
    ----------
    models_results : dict
        A dictionary where keys are model names (str) and values are dictionaries
        that must contain a 'predictions' key (array-like) with predicted class labels.
        Example:
        {
            "Logistic Regression": {"predictions": [...], ...},
            "Random Forest": {"predictions": [...], ...},
            ...
        }
    y_test : array-like
        The true labels for the test set.
    figsize : tuple, optional
        Size of the entire figure grid. Default is (15, 12).

    Returns:
    -------
    None
        Displays a grid of confusion matrix heatmaps.
    """
    n_models = len(models_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for i, (model_name, results) in enumerate(models_results.items()):
        if i < len(axes):
            y_pred = results['predictions']
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {model_name}')
            axes[i].set_xlabel('Prediction')
            axes[i].set_ylabel('Real')

    # Hide any unused subplots
    for i in range(len(models_results), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_roc_curves(models_results, y_test, figsize=(10, 8)):
    """
    Plots ROC curves for multiple classification models to compare their performance.

    This function expects a dictionary of model results, where each value contains 
    a 'probabilities' key with predicted probabilities for the positive class.
    It calculates the False Positive Rate (FPR), True Positive Rate (TPR), and 
    Area Under the Curve (AUC) for each model and plots them.

    Parameters:
    ----------
    models_results : dict
        A dictionary where keys are model names (str) and values are dictionaries
        that must contain a 'probabilities' key (array-like) with predicted probabilities.
        Example:
        {
            "Logistic Regression": {"probabilities": [...], ...},
            "Random Forest": {"probabilities": [...], ...},
            ...
        }
    y_test : array-like
        The true binary labels for the test set.
    figsize : tuple, optional
        Size of the figure to plot. Default is (10, 8).

    Returns:
    -------
    None
        Displays the ROC curves plot.
    """
    plt.figure(figsize=figsize)
    for model_name, results in models_results.items():
        y_pred_proba = results.get('probabilities')
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positives Rate')
    plt.ylabel('True Positives Rate')
    plt.title('ROC Curve - Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def detect_outliers_iqr(df, columns):
    """
    Detects the number of outliers in specified numerical columns of a DataFrame using the IQR method.

    An outlier is defined as a data point that lies below Q1 - 1.5*IQR or above Q3 + 1.5*IQR,
    where Q1 and Q3 are the 25th and 75th percentiles, respectively.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    columns : list of str
        A list of column names (numerical) to check for outliers.

    Returns:
    -------
    dict
        A dictionary where keys are column names and values are the number of detected outliers in each column.
    """
    outliers_count = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_count[col] = len(outliers)
    return outliers_count

def erase_outliers_iqr(df, columns):
    """
    Removes rows from the DataFrame that contain outliers in the specified columns using the IQR method.

    For each column, an outlier is any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR,
    where Q1 and Q3 are the 25th and 75th percentiles, respectively. The function iteratively filters
    out outliers column by column.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    columns : list of str
        A list of column names (numerical) from which outliers should be removed.

    Returns:
    -------
    pandas.DataFrame
        A filtered DataFrame with outliers removed from the specified columns.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df