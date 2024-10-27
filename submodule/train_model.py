import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

def train_model(model_type, data, use_tuned=False):
    # Scaling and Encoding Features
    df1 = data.copy()
    cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
    con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

    # Encode categorical columns and scale continuous features
    df1 = pd.get_dummies(df1, columns=cat_cols, drop_first=True)
    X = df1.drop(['output'], axis=1)
    y = df1['output']
    scaler = RobustScaler()
    X[con_cols] = scaler.fit_transform(X[con_cols])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'SVM':
        if use_tuned:
            # Hyperparameter-tuned SVM
            svm = SVC()
            parameters = {"C": np.arange(1, 10, 1),
                          'gamma': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]}
            searcher = GridSearchCV(svm, parameters)
            searcher.fit(X_train, y_train)
            model = searcher.best_estimator_
        else:
            # Untuned SVM
            model = SVC(kernel='linear', C=1, random_state=42, probability=True)
    
    elif model_type == 'LogisticRegression':
        model = LogisticRegression()
    
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=42)
    
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(n_estimators=300, max_depth=1, subsample=0.8, max_features=0.2, random_state=42)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Generate confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="coolwarm")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'{model_type} Confusion Matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes'])
    ax.yaxis.set_ticklabels(['No', 'Yes'])

    # ROC Curve only for Logistic Regression
    fig_roc = None
    if model_type == 'LogisticRegression':
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.plot(fpr, tpr, label=model_type)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{model_type} ROC Curve")
    
    return model, accuracy, report, fig, fig_roc