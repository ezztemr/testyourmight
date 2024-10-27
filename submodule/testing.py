# Scaling
from sklearn.preprocessing import RobustScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Models
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve, confusion_matrix, precision_score, f1_score

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def train_model(model, data):
    #---------------------------Scaling and Encoding features-------------------------------
    
    # creating a copy of df
    df1 = data
    
    # define the columns to be encoded and scaled
    cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
    con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
    
    # encoding the categorical columns
    df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)
    
    # defining the features and target
    X = df1.drop(['output'],axis=1)
    y = df1[['output']]
    
    # instantiating the scaler
    scaler = RobustScaler()
    
    # scaling the continuous featuree
    X[con_cols] = scaler.fit_transform(X[con_cols])
    print("The first 5 rows of X are")
    X.head()

    #---------------------------------Train and Test Split----------------------------------
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    print("The shape of X_train is      ", X_train.shape)
    print("The shape of X_test is       ",X_test.shape)
    print("The shape of y_train is      ",y_train.shape)
    print("The shape of y_test is       ",y_test.shape)

    #---------------------------------Linear Classifier; SVM-------------------------------

    clf = SVC(kernel='linear', C=1, random_state=42).fit(X_train,y_train)
    
    # predicting the values
    y_pred = clf.predict(X_test)
    
    # printing the test accuracy
    print("The test accuracy score of SVM is ", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_pred, y_test)
    f, ax= plt.subplots(1,1,figsize=(5,3))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels') ; ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes']) ; ax.yaxis.set_ticklabels(['No', 'Yes'])

    print(classification_report(y_test, y_pred))

    #Hyperparameter tuning for SVC

    # instantiating the object
    svm = SVC()
    
    # setting a grid - not so extensive
    parameters = {"C":np.arange(1,10,1),
                  'gamma':[0.00001,0.00005, 0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5]}
    
    # instantiating the GridSearchCV object
    searcher = GridSearchCV(svm, parameters)
    
    # fitting the object
    searcher.fit(X_train, y_train)
    
    # the scores
    print("The best params are :", searcher.best_params_)
    print("The best score is   :", searcher.best_score_)
    
    # predicting the values
    y_pred = searcher.predict(X_test)
    
    # printing the test accuracy
    print("The test accuracy score of SVM after hyper-parameter tuning is ", accuracy_score(y_test, y_pred))

    #--------------------------Logistic Regression--------------------------------------

    # instantiating the object
    logreg = LogisticRegression()
    
    # fitting the object
    logreg.fit(X_train, y_train)
    
    # calculating the probabilities
    y_pred_proba = logreg.predict_proba(X_test)
    
    # finding the predicted valued
    y_pred = np.argmax(y_pred_proba,axis=1)
    
    # printing the test accuracy
    print("The test accuracy score of Logistric Regression is ", accuracy_score(y_test, y_pred))
    
    cm = confusion_matrix(y_pred, y_test)
    f, ax= plt.subplots(1,1,figsize=(5,3))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels') ; ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes']) ; ax.yaxis.set_ticklabels(['No', 'Yes'])

    print(classification_report(y_test, y_pred))

    #Plotting Logistic Regression ROC Curve

    # calculating the probabilities
    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    
    # instantiating the roc_cruve
    fpr,tpr,threshols=roc_curve(y_test,y_pred_prob)
    
    # plotting the curve
    plt.plot([0,1],[0,1],"k--",'r+')
    plt.plot(fpr,tpr,label='Logistic Regression')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistric Regression ROC Curve")
    plt.show()

    #----------------------------------Tree Models; Decision Tree----------------------------

    # instantiating the object
    dt = DecisionTreeClassifier(random_state = 42)
    
    # fitting the model
    dt.fit(X_train, y_train)
    
    # calculating the predictions
    y_pred = dt.predict(X_test)
    
    # printing the test accuracy
    print("The test accuracy score of Decision Tree is ", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_pred, y_test)
    f, ax= plt.subplots(1,1,figsize=(5,3))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels') ; ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes']) ; ax.yaxis.set_ticklabels(['No', 'Yes'])

    print(classification_report(y_test, y_pred))

    #---------------------------Tree Models; Random Forest----------------------------------

    # instantiating the object
    rf = RandomForestClassifier()
    
    # fitting the model
    rf.fit(X_train, y_train)
    
    # calculating the predictions
    y_pred = rf.predict(X_test)
    
    # printing the test accuracy
    print("The test accuracy score of Random Forest is ", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_pred, y_test)
    f, ax= plt.subplots(1,1,figsize=(5,3))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels') ; ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes']) ; ax.yaxis.set_ticklabels(['No', 'Yes'])

    print(classification_report(y_test, y_pred))

    #--------------------Gradient Boosting Classifier - without tuning----------------------
    
    # instantiate the classifier
    gbt = GradientBoostingClassifier(n_estimators = 300,max_depth=1,subsample=0.8,max_features=0.2,random_state=42)
    
    # fitting the model
    gbt.fit(X_train,y_train)
    
    # predicting values
    y_pred = gbt.predict(X_test)
    print("The test accuracy score of Gradient Boosting Classifier is ", accuracy_score(y_test, y_pred))
    
    cm = confusion_matrix(y_pred, y_test)
    f, ax= plt.subplots(1,1,figsize=(5,3))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels') ; ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes']) ; ax.yaxis.set_ticklabels(['No', 'Yes'])
    
    print(classification_report(y_test, y_pred))

    return model, score