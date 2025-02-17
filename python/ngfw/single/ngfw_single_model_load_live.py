import asyncio
import pandas as pd
import joblib
from keras.models import load_model
import csv
# Single Record models load live

# Define the features
features20 = ['ct_state_ttl', 'sload', 'rate', 'sttl', 'smean', 'dload', 'sbytes', 'ct_srv_dst', 'ct_dst_src_ltm', 'dbytes', 'ackdat', 'dttl', 'ct_dst_sport_ltm', 'dmean','ct_srv_src', 'dinpkt', 'tcprtt', 'dur', 'synack', 'sinpkt']

features40 = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

current_folder = "d:\\miniconda\\UNSW-NB15\\ngfw"
slash="\\"
split="_90_5_5b_"

datapath=current_folder +slash + "data" + slash
f_test=datapath + 'unsw-nb15_testing' + split+ '.csv'

modelpath=current_folder + slash + "models" + slash

log_path = current_folder +slash + "results" + slash +'log_single.csv'
log_path2= current_folder +slash + "results" + slash +'log_single2.csv'

import asyncio
import threading
import psutil
import os
#from datetime import datetime
import random
import datetime
import time

#features20
#features40
#load df_test 
#X_test, y_test must be global

# Standard classification metrics
total_samples = 0          # Total number of records processed
true_positives = 0         # Correctly classified attack records
false_positives = 0        # Benign records misclassified as attacks
true_negatives = 0         # Correctly classified benign records
false_negatives = 0        # Attack records misclassified as benign

# Calculated evaluation metrics
accuracy = 0.0             # Overall classification accuracy
precision = 0.0            # TP / (TP + FP)
recall = 0.0               # TP / (TP + FN)
f1_score = 0.0             # 2 * (Precision * Recall) / (Precision + 

# Model adaptation tracking
model_switch_count = 0      # Number of model switches
model_switch_time = {}     # List of switch times (ms)
inference_speed = []        # List of records processed per second
scaler_transition_errors = 0 # Number of incorrect scaler transitions

testing_on=0
actual_attack = 0
predicted_attack_as_attack = 0
predicted_attack_as_benign = 0
actual_benign = 0
predicted_benign_as_benign = 0
predicted_benign_as_attack = 0

index20 = 0
index40 = 0

max_rec =0
# System resource monitoring
#cpu_usage = []       # List of CPU usage values over time
#memory_usage = []    # List of memory usage values over time
#latency = []         # List of latency values (ms) for each record classification

# Attack detection log
#attack_log = []  # Store detected attack records for later analysis

debug_str="start"
testing_time=[] 

def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S.%f")# [:-3] Trims

def append_log2(count, model, speed, xtime):
    global log_path2 
    unix_time = int(time.time())
    with open(log_path2,mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([unix_time,count, model, speed, xtime])
        
def append_log(count, model, speed, xtime):
    global log_path 
    unix_time = int(time.time())
    with open(log_path,mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([unix_time,count, model, speed, xtime])

  

def create_scaler(features, fnx):
    global df_test, s_next_model
    df_sample = df_test.sample(1000, random_state=42) 
    X_test = df_sample[features].values  
    y_test = df_sample['label'].values
    scaler_path = modelpath + 'scaler_' + fnx + '.pkl'
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(X_test)
    return scaler 

def load_scaler_data(s_next_model):
    global df_test #, modelpath, features20, features40
    global caler, n_samples
    global scaler20, scaler40
    
    ts=get_time()  # Assuming get_current_time() is defined
    print(ts, "Loading scaler data for", s_next_model)
    
    if s_next_model[-2:] == "20":
        scaler = scaler20  
    else:
        scaler = scaler40 
    return scaler 

def load_test_data(n):
    global df_test, X_test20, X_test40, y_test, scaler20, scaler40
    global features20, features40, max_rec
    df_test_n = df_test.iloc[:n]  
    X_test20 = df_test_n[features20].values  
    y_test = df_test_n['label'].values
    max_rec =len(y_test) -1  
    X_test20 = scaler20.transform(X_test20)
    
    X_test40 = df_test_n[features40].values  
    #y_test = df_test['label'].values
    X_test40 = scaler40.transform(X_test40)
    

def switch_scaler_data(s_next_model):
    global scalar
    ts=get_time()   
    
    if (s_next_model[-2:] == s_running_model[-2:]):
        print(ts, "Skipping scalar data", s_next_model[-2:])
        pass
    else:
        #print(ts, "Changing Scalar data", s_next_model[-2:])
        load_scaler_data(s_next_model)
    return
    
# Global variables
def initialize():
    global df_test, f_test
    global load, speed, speed_changed, sd_models
    global s_running_model, s_next_model, active_model, b_model_loading, b_model_active
    global t_send_data, t_inference_rate, t_test_rate
    global scaler20, scaler40
    global debug_str
    
    b_model_loading=1
    t_send_data=0.2 #1
    t_inference_rate=4#
    t_test_rate=0.5
    n_test=1000
    
    ts=get_time()
    print(ts, "Initializing .....")
    
    n_samples=50
    load = 1  # Number of records
    speed = 1  # Speed of sending
    speed_changed = 0

    # Model switching
    sd_models = ["", "deep_40", "shallow_40", "deep_20", "shallow_20"]
    debug_str="initialize()"
    s_running_model = ""
    s_next_model = sd_models[speed]
    b_model_active=0
    df_test = pd.read_csv(f_test)
    scaler20=create_scaler(features20, "20")
    scaler40=create_scaler(features40, "40")
    load_test_data(n_test)
    activate_model()

      
def switch_model(s_next_model):
    global modelpath, b_model_loading, s_running_model, b_model_active, debug_str, speed
    global model_switch_count, active_model
    t1 = time.time()  # Record start time
    ts=get_time()
    print(debug_str, "---------yyyyy")
    print(ts, "Loading model", s_next_model)
    b_model_loading=1
    switch_scaler_data(s_next_model)
    model_path=modelpath+ s_next_model + '_model_ANN5_hp.keras'
    
    t1 = time.time()  # Record start time
    active_model = load_model(model_path)
    
    #sleep_time=random.uniform(0, 2)
    #time.sleep(sleep_time)
    
    t2 = time.time()  
    # Store switch details in dictionary
    # Increment switch count
    model_switch_count += 1
    model_switch_time[t2] = {
    "count": model_switch_count,
    "model": s_next_model,
    "speed": speed,  
    "time": t2 - t1
    } 
    append_log(model_switch_count, s_next_model, speed, t2 - t1)
    #print(model_path)
    print (model_switch_time[t2])
    ts=get_time()
    b_model_loading=0
    print(ts, "Loaded", s_next_model)
    s_running_model=s_next_model
    #b_model_active=1

def activate_model():
    global speed, speed_changed, s_running_model, b_model_loading, b_model_active, s_next_model
    ts=get_time()
    print(ts, "Running" , s_running_model)
    if (s_running_model != s_next_model):
        switch_model(s_next_model)

async def test_data():
    #test single record by record
    global b_model_loading, load, speed, s_running_model
    global X_test20, X_test40, y_test, active_model
    global actual_attack, predicted_attack_as_attack, predicted_attack_as_benign
    global actual_benign, predicted_benign_as_benign, predicted_benign_as_attack, t_test_rate
    global index20, index40 
    global log_path2
    
    if b_model_loading == 0:  # Model is loaded
        if s_running_model[-2:] == "20":
            if index20 < max_rec:
                print(s_running_model[-2:],index20)
                r = X_test20[index20].reshape(1, -1)
                t1 = time.time()
                y_pred = active_model.predict(r, verbose=0)[0]  # 
                t2 = time.time()
                append_log2(s_running_model, index20,speed, t2 - t1)
                index20 += 1 
                if y_test[index20] == 1:
                    actual_attack += 1
                    if y_pred == 1:
                        predicted_attack_as_attack += 1
                    else:
                        predicted_attack_as_benign += 1
                else:
                    actual_benign += 1
                    if y_pred == 0:
                        predicted_benign_as_benign += 1
                    else:
                        predicted_benign_as_attack += 1
                       
        elif s_running_model[-2:] == "40":
            if index40 < max_rec:
                print(s_running_model[-2:], index40)
                r = X_test40[index40].reshape(1, -1)
                t1 = time.time()
                y_pred = active_model.predict(r, verbose=0)[0]   
                t2 = time.time()
                append_log2(s_running_model, index40,speed, t2 - t1)
                index40 += 1
                if y_test[index40] == 1:
                    actual_attack += 1
                    if y_pred == 1:
                        predicted_attack_as_attack += 1
                    else:
                        predicted_attack_as_benign += 1
                else:
                    actual_benign += 1
                    if y_pred == 0:
                        predicted_benign_as_benign += 1
                    else:
                        predicted_benign_as_attack += 1
                

    if (index20 == max_rec) and (index40 == max_rec):
        # Calculate Metrics
        TP = predicted_attack_as_attack  # True Positives
        FP = predicted_benign_as_attack  # False Positives
        FN = predicted_attack_as_benign  # False Negatives
        TN = predicted_benign_as_benign  # True Negatives

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Print results
        print("Actual Attack:", actual_attack)
        print("Predicted Attack as Attack (TP):", predicted_attack_as_attack)
        print("Predicted Attack as Benign (FN):", predicted_attack_as_benign)
        print("Actual Benign:", actual_benign)
        print("Predicted Benign as Benign (TN):", predicted_benign_as_benign)
        print("Predicted Benign as Attack (FP):", predicted_benign_as_attack)

        print("\nClassification Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")
        exit()
    await asyncio.sleep(t_test_rate) #4
   
async def change_inference_rate():
    global t_inference_rate
    global speed, speed_changed, s_running_model, s_next_model, b_model_loading
    global debug_str, sd_models
    if b_model_loading == 0:
        debug_str = "change_inference_rate()"  
        num = random.randint(1, 100)  # random integer
        if num % 5 == 0:  # Check if num is divisible by 5
            load = 1  # Number of records
            speed = random.randint(1, 100) % 4 + 1  # Some speed factor
            s_next_model = sd_models[speed]  # Store next model
            ts = get_time()  # Ensure get_time() function exists
            print(ts, f"Speed changed to {speed} =============")
            print(ts, f"Switching to model {speed}: {sd_models[speed]} ------")

            activate_model() 
    else:
        pass
        #print("---")
    # loop doesn't block execution
    #print("b_model_loading", b_model_loading)
    await asyncio.sleep(t_inference_rate) #4
    
async def xtest():
    print("xtest")
    await asyncio.sleep(4)

async def main():
    
    while True:  # Infinite loop
        await asyncio.gather(
            change_inference_rate(),
            test_data(),
            #xtest(),
            #test_data(),
            
        )
        await asyncio.sleep(1)  # Optional: Adds a small delay between iterations for better control

initialize()
# Run the async event loop
asyncio.run(main())

