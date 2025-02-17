import psutil
import time
import os
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import tensorflow as tf
import numpy as np
'''
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2
#from kerastuner import HyperParameters
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
'''

def performance_data(y_test, y_pred_classes):
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

    print("----------------------------------")

xmodels=["shallow_20", "deep_20", "shallow_40", "deep_40"]


features20 = ['dur', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sbytes', 'dbytes', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_src_ltm', 'ct_dst_sport_ltm','ct_srv_dst']
              
features40 = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

# Define the features
features=[features20, features40]  # List used as an array
slash="\\"
split="_90_5_5b_"

current_folder = "d:\\miniconda\\UNSW-NB15\\ngfw"
print(current_folder)
slash="\\"
split="_90_5_5b_"
datapath=current_folder +slash + "data" + slash
modelpath=current_folder + slash + "models" + slash

#data_folder = "d:\\miniconda\\UNSW-NB15\\testing"
#current_folder = "d:\\miniconda\\UNSW-NB15\\ngfw"
#results_folder=current_folder+ slash + "results"


#datapath=data_folder +slash + "data" + slash
modelpath=current_folder + slash + "models" + slash
print(modelpath)

f_train=datapath+'unsw-nb15_training' + split +'.csv'
print(f_train)

f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'
print(f_test)

f_performance_file=current_folder +slash + "results" + slash +  'result.txt'
print(f_performance_file)

df_train = pd.read_csv(f_train)
df_test = pd.read_csv(f_test)

xfeatures=features20

X_train = df_train[features20].values 
y_train= df_train['label'].values # 'label' contains labels

# Separate features and labels for testing set
X_test = df_test[features20].values  
y_test= df_test['label'].values
    
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#X_val = scaler.transform(X_val)

xmodel="shallow_20"
model_path=modelpath+ xmodel+ '_model_ANN5_hp.keras'
print(model_path)

saved_model = load_model(model_path)  
y_pred = saved_model.predict(X_test)
# Convert predicted probabilities to binary predictions (0 or 1)
y_pred_classes = (y_pred > 0.5).astype(int)
performance_data(y_test, y_pred_classes)
'''

def create_testset(df_train, df_test, xfeatures):
    # Separate features and labels for training set
    X_train = df_train[xfeatures].values # 'label' contains labels
    y_train= df_train['label'].values
    # Separate features and labels for testing set
    X_test = df_test[xfeatures].values  
    y_test= df_test['label'].values
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_test, y_test

X_test20, y_test20 = create_testset(df_train,df_test, features20)
X_test40, y_test40 = create_testset(df_train, df_test, features40)

for xmodel in xmodels:
    model_path=modelpath+ xmodel+ '_model_ANN5_hp.keras'        
    print(model_path)
    if "20" in xmodel:
        X_test=X_test20
        y_test=y_test20
    else:
        X_test=X_test40
        y_test=y_test40
    print("n featurs", X_test.shape[1])
    
    # Measure CPU and memory usage before prediction
    cpu_usage_before = psutil.cpu_percent()
    memory_usage_before = psutil.virtual_memory().used
    # Record start time for prediction
    start_time = time.time()
    # Load the saved model
    active_model = load_model(model_path)  
    # Record end time for prediction
    end_time = time.time()
    # Measure CPU and memory usage after prediction
    cpu_usage_after = psutil.cpu_percent()
    memory_usage_after = psutil.virtual_memory().used

    # Print CPU and memory usage, and prediction time
    print("CPU usage:", cpu_usage_after-cpu_usage_before)
    print("Memory usage:", memory_usage_after- memory_usage_before)
    print("Loading time:", end_time - start_time)
    print("-------------------------------------")

    # Record start time for prediction
    start_time = time.time()
    # Predict on the test data  
    y_pred = active_model.predict(X_test)
    # Convert predicted probabilities to binary predictions (0 or 1)
    y_pred_classes = (y_pred > 0.5).astype(int)
    # Record end time for prediction
    end_time = time.time()

    # Measure CPU and memory usage after prediction
    cpu_usage_after = psutil.cpu_percent()
    memory_usage_after = psutil.virtual_memory().used

    # Calculate prediction time
    prediction_time = end_time - start_time
    
     # Print CPU and memory usage, and prediction time
    print("CPU usage:", cpu_usage_after-cpu_usage_before)
    print("Memory usage:", memory_usage_after- memory_usage_before)
    print("Loading time:", end_time - start_time)
    print("-------------------------------------")

'''
