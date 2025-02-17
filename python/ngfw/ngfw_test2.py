import psutil
import time
import os
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import pandas as pd
import tensorflow as tf

# Model list
xmodels = ["shallow_20", "deep_20", "shallow_40", "deep_40"]

# Define features
features20 = ['dur', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sbytes', 'dbytes', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_src_ltm', 'ct_dst_sport_ltm','ct_srv_dst']
features40 = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

# Paths
data_folder = "d:\\miniconda\\UNSW-NB15\\testing"
current_folder = "d:\\miniconda\\UNSW-NB15\\ngfw"
results_folder = os.path.join(current_folder, "results")
datapath = os.path.join(data_folder, "data")
modelpath = os.path.join(current_folder, "models")

f_train = os.path.join(datapath, 'unsw-nb15_training_90_5_5b_.csv')
f_test = os.path.join(datapath, 'unsw-nb15_testing_90_5_5b_.csv')
f_performance_file = os.path.join(results_folder, 'results.txt')

# Load datasets
df_train = pd.read_csv(f_train)
df_test = pd.read_csv(f_test)

# Standardize training data
scaler = StandardScaler()
scaler.fit(df_train[features40])  # Fit on training data

def create_testset(df, xfeatures, scaler):
    """Prepares test dataset."""
    X_test = scaler.transform(df[xfeatures].values)  # Transform test set
    y_test = df['label'].values
    return X_test, y_test

X_test20, y_test20 = create_testset(df_test, features20, scaler)
X_test40, y_test40 = create_testset(df_test, features40, scaler)

# Model Evaluation Loop
for xmodel in xmodels:
    model_path = os.path.join(modelpath, f"{xmodel}_model_ANN5_hp.keras")
    
    if "20" in xmodel:
        X_test, y_test = X_test20, y_test20
    else:
        X_test, y_test = X_test40, y_test40

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        continue

    # Load the saved model
    start_time = time.time()
    active_model = load_model(model_path)
    load_time = time.time() - start_time
    
    # Measure CPU & memory before prediction
    cpu_before = psutil.cpu_percent()
    memory_before = psutil.virtual_memory().used

    # Make predictions
    start_time = time.time()
    y_pred = active_model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)  # Convert to binary
    prediction_time = time.time() - start_time

    # Measure CPU & memory after prediction
    cpu_after = psutil.cpu_percent()
    memory_after = psutil.virtual_memory().used

    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, zero_division=1)
    recall = recall_score(y_test, y_pred_classes, zero_division=1)
    f1 = f1_score(y_test, y_pred_classes)
    roc_auc = roc_auc_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred)

    # Confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()

    # Print results
    print(f"Model: {xmodel}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Prediction Time: {prediction_time:.4f} sec")
    print(f"CPU Usage Change: {cpu_after - cpu_before}%")
    print(f"Memory Usage Change: {(memory_after - memory_before) / (1024 ** 2):.2f} MB")
    print("--------------------------------------------------")
