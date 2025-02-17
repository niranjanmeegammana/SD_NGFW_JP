'''
Output:
Memory usage before and after loading each model.
Memory usage after all models are loaded into memory.
Memory usage after cleaning up (unloading) the models.
'''
import psutil
import os
import tensorflow as tf
import gc
import time
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

sd_models = ["", "deep_40", "shallow_40", "deep_20", "shallow_20"]

def check_model_memory1(model_name):
    xpath = r"D:\miniconda\UNSW-NB15\ngfw\models\\"  # Directory path
    model_path = xpath + model_name + '_model_ANN5_hp.keras'
    
    # Record the timestamp (ts)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(ts, "Loading model", model_name)

    # Get the current process ID
    pid = os.getpid()

    # Get the process object using psutil
    process = psutil.Process(pid)

    # Get memory usage before loading the model (in MB)
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"Memory before loading {model_name}: {memory_before:.2f} MB")

    # Load the model
    model = load_model(model_path)

    # Garbage collection before measuring memory
    gc.collect()

    # Get memory usage after loading the model (in MB)
    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"Memory after garbage collection: {memory_after:.2f} MB")

    memory_used_by_model = memory_after - memory_before
    print(f"Memory used by {model_name}: {memory_used_by_model:.2f} MB")
    
    return model

def check_model_memory2(model_name):
    xpath = r"D:\miniconda\UNSW-NB15\ngfw\models\\"  # Directory path
    model_path = xpath + model_name + '_model_ANN5_hp.keras'

    # Load the model
    model = load_model(model_path)
    return model

def check_tensorflow_memory():
    # Checking memory usage through TensorFlow if available
    if tf.config.list_physical_devices('GPU'):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f"TensorFlow GPU memory info: {memory_info}")
    else:
        print("No GPU detected or not available.")

def clear_tensorflow_session():
    # Clear the session to release memory
    K.clear_session()
    print("TensorFlow session cleared")

# Iterate through the models and check memory usage
for i in range(4):
    model_name = sd_models[i + 1]  # Get model name
    check_model_memory1(model_name)  # Check memory for the model

print("-------------------------")
# Record the timestamp (ts)
ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(ts, "Loading models")

# Get the current process ID
pid = os.getpid()

# Get the process object using psutil
process = psutil.Process(pid)

# Load all models and check memory usage
deep_40 = check_model_memory2("deep_40")
shallow_40 = check_model_memory2("shallow_40")
deep_20 = check_model_memory2("deep_20")
shallow_20 = check_model_memory2("shallow_20")

# Check memory usage after loading all models
memory_after_loading_all = process.memory_info().rss / (1024 * 1024)  # MB
print(f"Memory usage after loading all models: {memory_after_loading_all:.2f} MB")

# Garbage collection before unloading models
gc.collect()

# Clear TensorFlow session and check memory usage after unloading models
clear_tensorflow_session()

# Check memory after clearing session
memory_after_unloading_all = process.memory_info().rss / (1024 * 1024)  # MB
print(f"Memory usage after unloading models: {memory_after_unloading_all:.2f} MB")

# Check TensorFlow's internal memory usage
check_tensorflow_memory()
