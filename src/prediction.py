import numpy as np

# Function to predict a single instance
def predict_single_instance(model, input_data):
    input_data_np = np.array(input_data).reshape(1, -1)  
    prediction = model.predict(input_data_np)[0]  
    return "Malignant" if prediction == 0 else "Benign" 
