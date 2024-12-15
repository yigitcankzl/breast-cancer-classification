import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to load and prepare the dataset
def load_and_prepare_data():
    dataset = sklearn.datasets.load_breast_cancer() 
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    data['label'] = dataset.target 
    return data

# Function to split the dataset into training and testing sets
def split_data(data, test_size=0.2, random_state=2):
    X = data.drop(columns='label', axis=1)  
    Y = data['label']  
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

# Function to scale the features
def scale_data(X_train, X_test):
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)  
    return X_train_scaled, X_test_scaled
