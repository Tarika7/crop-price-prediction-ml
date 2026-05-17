import pickle
import numpy as np
import os

# Check if models folder exists
if not os.path.exists('../models'):
    print("ERROR: '../models' folder not found!")
    print("Current directory:", os.getcwd())
    print("Create the 'models' folder or run from 'src' directory")
else:
    # Test Tamil Nadu rice model
    with open('../models/tn_rice_model.pkl', 'rb') as f:
        rice_model = pickle.load(f)
    
    sample = np.array([[2025, 10, 3, 2750, 2850, 100, 2700, 2800, 90]])
    print("TN Rice Prediction:", rice_model.predict(sample))
