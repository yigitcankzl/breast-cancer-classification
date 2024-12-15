from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Function to train Logistic Regression model
def train_logistic_regression(X_train, Y_train):
    model = LogisticRegression(max_iter=5000, random_state=2, C=0.1)  
    model.fit(X_train, Y_train) 
    return model

# Function to train Random Forest model
def train_random_forest(X_train, Y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2) 
    model.fit(X_train, Y_train)  
    return model

# Function to train Support Vector Machine model
def train_svm(X_train, Y_train):
    model = SVC(kernel='linear', random_state=2)  
    model.fit(X_train, Y_train)  
    return model
