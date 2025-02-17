import psutil
import time
import joblib
from datetime import datetime
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
#print(current_folder)
slash="\\"
split="_90_5_5b_"
datapath=current_folder +slash + "data" + slash
modelpath=current_folder + slash + "models" + slash

f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'

f_performance_file=current_folder +slash + "results" + slash + 'results.txt'

#print(f_test)

df_test = pd.read_csv(f_test)

def load_scaler_data(df_test, modelpath, xmodel,features,n):
    # Randomly select n rows
    df_sample = df_test.sample(n, random_state=42)  
    X_test = df_sample[features].values  
    y_test = df_sample['label'].values  
    #X_test = df_test[features].values
    #y_test = df_test['label'].values
    scaler_path = modelpath + 'scaler_' + xmodel[-2:] + '.pkl'
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(X_test)
    return X_test,y_test, scaler 


xmodels=["shallow_20", "deep_20", "shallow_40", "deep_40"]
#xmodel="shallow_20"

done_20 =False
done_40 =False

for xmodel in xmodels:
    if "20" in xmodel:
        if not (done_20):
           X_test,y_test, scaler = load_scaler_data(df_test, modelpath, xmodel,features20, 9000)
           done_20=True
    else:
        if not(done_40):
            X_test,y_test, scaler = load_scaler_data(df_test, modelpath, xmodel,features40, 9000)
            done_40=True
    

    model_path=modelpath+ xmodel+ '_model_ANN5_hp.keras'
    active_model = load_model(model_path)
    pred=[0,0]
    actual=[0,0]
    check=[0,0]
    m=0
    for r in X_test:
        #print(r)
        r = r.reshape(1, -1)
        y_pred = active_model.predict(r,  verbose=0)
        # Convert predicted probabilities to binary predictions (0 or 1)
        y_pred_class = (y_pred > 0.5).astype(int)
        #print(y_pred_class)
        
        if (y_test[m]==1):
            actual[0]=actual[0]+1
        else:
            actual[1]=actual[1]+1
        
        if (y_pred_class==1):
            pred[0]=pred[0]+1
        else:
            pred[1]=pred[1]+1
        
        if (y_test[m]==y_pred_class):
            check[0]=check[0]+1
        else:
            check[1]=check[1]+1    
        m=m+1
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Date & Time:", current_datetime)
    print( xmodel)    
    print(actual)        
    print(pred)
    print(check)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Date & Time:", current_datetime)
    print("-------------------")