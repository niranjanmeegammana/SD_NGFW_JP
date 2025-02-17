import asyncio
import psutil
import time
import joblib
from datetime import datetime
from keras.models import load_model
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

async_stop_flag = False

# Define the features
features20 = ['ct_state_ttl', 'sload', 'rate', 'sttl', 'smean', 'dload', 'sbytes', 'ct_srv_dst', 'ct_dst_src_ltm', 'dbytes', 'ackdat', 'dttl', 'ct_dst_sport_ltm', 'dmean','ct_srv_src', 'dinpkt', 'tcprtt', 'dur', 'synack', 'sinpkt']

features40 = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

current_folder = "d:\\miniconda\\UNSW-NB15\\ngfw"
slash = "\\"
split = "_90_5_5b_"
datapath = current_folder + slash + "data" + slash
modelpath = current_folder + slash + "models" + slash

f_test = datapath + 'unsw-nb15_testing' + split + '.csv'
f_performance_file = current_folder + slash + "results" + slash + 'results.txt'

df_test = pd.read_csv(f_test)

def load_scaler_data(df_test, modelpath, xmodel, features, n):
    # Randomly select n rows
    df_sample = df_test.sample(n, random_state=42)  
    X_test = df_sample[features].values  
    y_test = df_sample['label'].values  
    scaler_path = modelpath + 'scaler_' + xmodel[-2:] + '.pkl'
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(X_test)
    return X_test, y_test, scaler 

xmodels = ["shallow_20", "deep_20", "shallow_40", "deep_40"]

# Run models asynchronously
async def run_models():
    done_20 = False
    done_40 = False
    for xmodel in xmodels:
        if "20" in xmodel and not done_20:
            X_test, y_test, scaler = load_scaler_data(df_test, modelpath, xmodel, features20, 100)
            done_20 = True
        elif "40" in xmodel and not done_40:
            X_test, y_test, scaler = load_scaler_data(df_test, modelpath, xmodel, features40, 100)
            done_40 = True
        
        model_path = modelpath + xmodel + '_model_ANN5_hp.keras'
        active_model = load_model(model_path)
        pred = [0, 0]
        actual = [0, 0]
        check = [0, 0]
        
        # Predicting in batches instead of one sample at a time
        batch_size = 32
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            batch_pred = active_model.predict(batch_X, verbose=0)
            batch_pred_class = (batch_pred > 0.5).astype(int)
            
            # Update counts based on predictions
            for j in range(len(batch_y)):
                if batch_y[j] == 1:
                    actual[0] += 1
                else:
                    actual[1] += 1
                
                if batch_pred_class[j] == 1:
                    pred[0] += 1
                else:
                    pred[1] += 1
                
                if batch_y[j] == batch_pred_class[j]:
                    check[0] += 1
                else:
                    check[1] += 1
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current Date & Time: {current_datetime}")
        print(xmodel)    
        print(actual)        
        print(pred)
        print(check)
        print("-------------------")
  
async_stop_flag = True  # stop the monitoring loop  

# Function to monitor process activity (CPU and memory usage)
async def monitor_process_activity():
    global async_stop_flag
    
    pid = os.getpid()  # Get the current process ID
    process = psutil.Process(pid)
    logf = current_folder + slash + "results" + slash + 'log.txt'
    
    # Open log file asynchronously
    async def async_write_log(message):
        with open(logf, "a") as log_file:  # Open in append mode
            log_file.write(message)
    
    while not async_stop_flag:
        # Get CPU usage as percentage and memory usage in MB
        cpu_usage = process.cpu_percent(interval=1)
        memory_usage = process.memory_info().rss / 1024 ** 2  # Memory in MB
        
        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare the log message with date and time
        log_message = f"[{current_time}], CPU: {cpu_usage}%, Memory: {memory_usage:.2f}MB\n"
        
        # Write the log message to the file asynchronously
        await async_write_log(log_message)
        
        # Print to console for immediate feedback (optional)
        #print(log_message.strip())
        
        # Simulate async waiting to avoid blocking the loop
        await asyncio.sleep(1)

# Main function to run all tasks concurrently
async def main():
    # Run all the functions concurrently using asyncio.gather
    await asyncio.gather(
        run_models(),
        monitor_process_activity()  # Monitor in parallel
    )

# Start the asyncio event loop
asyncio.run(main())
