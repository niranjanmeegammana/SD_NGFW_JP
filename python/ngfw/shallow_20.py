import psutil
import time
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

xmodel="shallow_20"
# Define the features
features = ['ct_state_ttl', 'sload', 'rate', 'sttl', 'smean', 'dload', 'sbytes', 'ct_srv_dst', 'ct_dst_src_ltm', 'dbytes', 'ackdat', 'dttl', 'ct_dst_sport_ltm', 'dmean','ct_srv_src', 'dinpkt', 'tcprtt', 'dur', 'synack', 'sinpkt']

datapath = "d:\\miniconda\\UNSW-NB15\\testing\\data\\"
current_folder = "d:\\miniconda\\UNSW-NB15\\ngfw\\"
modelpath=current_folder +"models\\"
print(current_folder)

slash="\\"
split="_90_5_5b_"

f_train=datapath+'unsw-nb15_training' + split +'.csv'
f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'
f_val=datapath + 'unsw-nb15_validation' + split+ '.csv'

f_performance_file=current_folder + "results\\"+ xmodel+ '.txt'

model_path=modelpath+ xmodel+ '_model_ANN5_hp.keras'

print(f_test)
print(f_train)
print(f_val)
print(model_path)

df_train = pd.read_csv(f_train)
df_test = pd.read_csv(f_test)
df_val=pd.read_csv(f_val)
print(df_test.head(5))

# Separate features and labels for training set
X_train = df_train[features].values # assuming 'label' is the column containing labels
y_train = df_train['label'].values

# Separate features and labels for testing set
X_test = df_test[features].values  # assuming 'label' is the column containing labels
y_test = df_test['label'].values

# Separate features and labels for testing set
X_val = df_val[features].values  # assuming 'label' is the column containing labels
y_val = df_val['label'].values


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

#==================================

# Load the saved model
saved_model = load_model(model_path)

# Measure CPU and memory usage before prediction
cpu_usage_before = psutil.cpu_percent()
memory_usage_before = psutil.virtual_memory().used

# Record start time for prediction
start_time = time.time()

# Assuming X_test and y_test are your test data
# Predict on the test data
y_pred = saved_model.predict(X_test)

# Convert predicted probabilities to binary predictions (0 or 1)
y_pred_classes = (y_pred > 0.5).astype(int)

# Record end time for prediction
end_time = time.time()

# Measure CPU and memory usage after prediction
cpu_usage_after = psutil.cpu_percent()
memory_usage_after = psutil.virtual_memory().used

# Calculate prediction time
prediction_time = end_time - start_time

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)
roc_auc = roc_auc_score(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred)

# Compute confusion matrix
conf_matrix = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()

# Print performance metrics
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Average Precision:", avg_precision)
print("Confusion Matrix:\n", conf_matrix)

# Print CPU and memory usage, and prediction time
print("CPU usage before prediction:", cpu_usage_before)
print("Memory usage before prediction:", memory_usage_before)
print("CPU usage after prediction:", cpu_usage_after)
print("Memory usage after prediction:", memory_usage_after)
print("Prediction time:", prediction_time)

