import psutil
import time
import joblib
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
'''
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

def performance_data(cpu_usage, memory_usage, prediction_time, y_test, y_pred_classes):

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
    print("CPU usage:", cpu_usage)
    print("Memory usage:", memory_usage)
    print("================================")
    
# Define the features
features20 = ['ct_state_ttl', 'sload', 'rate', 'sttl', 'smean', 'dload', 'sbytes', 'ct_srv_dst', 'ct_dst_src_ltm', 'dbytes', 'ackdat', 'dttl', 'ct_dst_sport_ltm', 'dmean','ct_srv_src', 'dinpkt', 'tcprtt', 'dur', 'synack', 'sinpkt']

features40 = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

current_folder = "d:\\miniconda\\UNSW-NB15\\ngfw"
print(current_folder)
slash="\\"
split="_90_5_5b_"
datapath=current_folder +slash + "data" + slash
modelpath=current_folder + slash + "models" + slash
f_train=datapath+'unsw-nb15_training' + split +'.csv'
f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'
#f_val=datapath + 'unsw-nb15_validation' + split+ '.csv'
f_performance_file=current_folder +slash + "results" + slash + 'results.txt'


print(f_test)
print(f_train)
#print(f_val)

df_train = pd.read_csv(f_train)
df_test = pd.read_csv(f_test)
#df_val=pd.read_csv(f_val)
#print(df_test.head(5))

def get_dataset(df_train,df_test, modelpath, xmodel, features):
    X_train = df_train[features].values  
    #y_train = df_train['label'].values
    X_test = df_test[features].values
    y_test = df_test['label'].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    scaler_path = modelpath + 'scaler_' + xmodel[-2:] + '.pkl'
    joblib.dump(scaler, scaler_path)
    loaded_scaler = joblib.load(scaler_path)
    X_test_scaled = loaded_scaler.transform(X_test)  
    return X_test, y_test

'''    
# Separate features and labels for training set
X_train = df_train[features20].values # assuming 'label' is the column containing labels
y_train = df_train['label'].values

# Separate features and labels for testing set
X_test = df_test[features20].values  # assuming 'label' is the column containing labels
y_test = df_test['label'].values

# Separate features and labels for testing set
#X_val = df_val[features].values  # assuming 'label' is the column containing labels
#y_val = df_val['label'].values

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#X_val = scaler.transform(X_val)
'''

xmodels=["shallow_20", "deep_20", "shallow_40", "deep_40"]

#X_test, y_test=get_dataset(df_train,df_test, features20)
done_20=False
done_40 =False
for xmodel in xmodels:
#xmodel="shallow_20"
    model_path=modelpath+ xmodel+ '_model_ANN5_hp.keras'
    if "20" in xmodel:
        if not (done_20):
            X_test, y_test=get_dataset(df_train,df_test,modelpath,  xmodel, features20)
            done_20=True
    else:
        if not(done_40):
            X_test, y_test=get_dataset(df_train,df_test, modelpath, xmodel, features40)
            done_40=True
            
    
    print(X_test.shape)
    print(y_test.shape)
    print(model_path)

    # Load the saved model
    active_model = load_model(model_path)

    # Measure CPU and memory usage before prediction
    cpu_usage_before = psutil.cpu_percent()
    memory_usage_before = psutil.virtual_memory().used

    # Record start time for prediction
    start_time = time.time()

    # Assuming X_test and y_test are your test data
    # Predict on the test data
    y_pred = active_model.predict(X_test)

    # Convert predicted probabilities to binary predictions (0 or 1)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Record end time for prediction
    end_time = time.time()

    # Measure CPU and memory usage after prediction
    cpu_usage_after = psutil.cpu_percent()
    memory_usage_after = psutil.virtual_memory().used

    cpu_usage=cpu_usage_after-cpu_usage_before
    memory_usage=memory_usage_after - memory_usage_before
    prediction_time = end_time - start_time

    performance_data(cpu_usage, memory_usage, prediction_time, y_test, y_pred_classes)

