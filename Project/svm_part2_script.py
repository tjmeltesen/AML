import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Set to True for training, False for test/inference
TRAINING = True  # Change to False for test/inference

dev_csv = 'PartII_dev.csv'  # Development set
test_csv = 'NEW_TEST_FILE.csv'  # Replace with actual test file when needed
model_path = 'svm_part2_model.joblib'

# Only use X1-X125 and Y
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    X = data[[f'X{i}' for i in range(1, 126)]]
    y = data['Y']
    return X, y

# Define pipeline and parameters
svm_params = {
    'C': 1,
    'kernel': 'rbf',
    'gamma': 'scale',
    'degree': 3,
    'decision_function_shape': 'ovr'
}
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(0.95)),
    ('svm', SVC(class_weight='balanced', **svm_params))
])

if TRAINING:
    X_dev, y_dev = load_data(dev_csv)
    model = pipe.fit(X_dev, y_dev)
    y_pred = model.predict(X_dev)
    print('SVM Training Performance:')
    print(classification_report(y_dev, y_pred))
    joblib.dump({'model': model, 'params': svm_params}, model_path)
    print(f'Model saved to {model_path}')
else:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file {model_path} not found! Train first.')
    saved = joblib.load(model_path)
    model = saved['model']
    # Use dev set for demonstration; replace with test_csv for real test
    X_test, y_test = load_data(dev_csv)
    y_pred = model.predict(X_test)
    print('SVM Test Performance:')
    print(classification_report(y_test, y_pred))
