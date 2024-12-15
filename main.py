from src.data_preprocessing import load_and_prepare_data, split_data, scale_data
from src.model_training import train_logistic_regression, train_random_forest, train_svm
from src.evaluation import evaluate_logistic_regression, evaluate_random_forest, evaluate_svm, evaluate_with_cross_validation
from src.prediction import predict_single_instance

def main():
    # Load and split the dataset
    data = load_and_prepare_data()
    X_train, X_test, Y_train, Y_test = split_data(data)

    # Scale the data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Train and evaluate Logistic Regression model
    logistic_model = train_logistic_regression(X_train_scaled, Y_train)
    evaluate_logistic_regression(logistic_model, X_train_scaled, Y_train, X_test_scaled, Y_test)
    evaluate_with_cross_validation(logistic_model, X_train_scaled, Y_train)

    # Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train_scaled, Y_train)
    evaluate_random_forest(rf_model, X_train_scaled, Y_train, X_test_scaled, Y_test)
    evaluate_with_cross_validation(rf_model, X_train_scaled, Y_train)

    # Train and evaluate SVM model
    svm_model = train_svm(X_train_scaled, Y_train)
    evaluate_svm(svm_model, X_train_scaled, Y_train, X_test_scaled, Y_test)
    evaluate_with_cross_validation(svm_model, X_train_scaled, Y_train)

    # Predict a single instance (example input data)
    input_data = (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 
                  0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
                  15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)

    # Print predictions for each model
    print("Logistic Regression Prediction:", predict_single_instance(logistic_model, input_data))
    print("Random Forest Prediction:", predict_single_instance(rf_model, input_data))
    print("SVM Prediction:", predict_single_instance(svm_model, input_data))

if __name__ == "__main__":
    main()
