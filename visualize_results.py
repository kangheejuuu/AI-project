# visualize_results.py

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report

def visualize_confusion_matrix(y_true, y_pred, class_names):
    """
    Visualizes the confusion matrix.

    Parameters:
    - y_true: List of true labels
    - y_pred: List of predicted labels
    - class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# visualize_results.py

def visualize_classification_report(class_report):
    """
    Visualizes the classification report.

    Parameters:
    - class_report: String containing the classification report
    """
    report_data = []
    lines = class_report.split('\n')
    for line in lines[2:-5]:
        row = line.split()
        if row:
            report_data.append(row)

    df = pd.DataFrame(report_data, columns=['precision', 'recall', 'f1-score'])
    df = df.astype({'precision': float, 'recall': float, 'f1-score': float})

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.set_index('support').T, annot=True, cmap='Blues')
    plt.title('Classification Report Heatmap')
    plt.show()


