from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Function to evaluate Logistic Regression model
def evaluate_logistic_regression(model, X_train, Y_train, X_test, Y_test):
    train_accuracy = accuracy_score(Y_train, model.predict(X_train))  
    test_accuracy = accuracy_score(Y_test, model.predict(X_test))  
    print("Logistic Regression:")
    print(f"  - Training Accuracy: {train_accuracy:.2f}")
    print(f"  - Test Accuracy: {test_accuracy:.2f}")
    return train_accuracy, test_accuracy

# Function to evaluate Random Forest model
def evaluate_random_forest(model, X_train, Y_train, X_test, Y_test):
    train_accuracy = accuracy_score(Y_train, model.predict(X_train)) 
    test_accuracy = accuracy_score(Y_test, model.predict(X_test))  
    print("Random Forest:")
    print(f"  - Training Accuracy: {train_accuracy:.2f}")
    print(f"  - Test Accuracy: {test_accuracy:.2f}")
    return train_accuracy, test_accuracy

# Function to evaluate SVM model
def evaluate_svm(model, X_train, Y_train, X_test, Y_test):
    train_accuracy = accuracy_score(Y_train, model.predict(X_train)) 
    test_accuracy = accuracy_score(Y_test, model.predict(X_test))  
    print("Support Vector Machine (SVM):")
    print(f"  - Training Accuracy: {train_accuracy:.2f}")
    print(f"  - Test Accuracy: {test_accuracy:.2f}")
    return train_accuracy, test_accuracy

# Function to perform cross-validation and display results
def evaluate_with_cross_validation(model, X_train, Y_train, cv=5):
    cv_scores = cross_val_score(model, X_train, Y_train, cv=cv)  
    print(f"CV Scores: {cv_scores}")  
    print(f"Mean CV Score: {np.mean(cv_scores):.2f}")  
