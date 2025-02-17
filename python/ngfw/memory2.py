import psutil
import os
import time
import gc
from tensorflow.keras.models import load_model
from prettytable import PrettyTable

sd_models = ["", "deep_40", "shallow_40", "deep_20", "shallow_20"]

def get_memory_usage():
    pid = os.getpid()
    process = psutil.Process(pid)
    return process.memory_info().rss / (1024 * 1024)  # MB

def check_model_memory(model_name):
    xpath = r"D:\miniconda\UNSW-NB15\ngfw\models\\"  # Directory path
    model_path = xpath + model_name + '_model_ANN5_hp.keras'
    
    # Record the timestamp (ts)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Before loading the model
    memory_before = get_memory_usage()

    print(ts, f"Loading model {model_name}")
    model = load_model(model_path)

    # Run garbage collection
    gc.collect()
    
    # After garbage collection
    memory_after_gc = get_memory_usage()

    # Memory used by the model
    memory_used = memory_after_gc - memory_before
    
    return memory_before, memory_after_gc, memory_used

# Create the table for memory usage
table = PrettyTable()
table.field_names = ["Model Name", "Memory Before Loading (MB)", "Memory After Garbage Collection (MB)", "Memory Used (MB)"]

# Collect memory data for each model
for model_name in sd_models[1:]:
    memory_before, memory_after_gc, memory_used = check_model_memory(model_name)
    table.add_row([model_name, f"{memory_before:.2f}", f"{memory_after_gc:.2f}", f"{memory_used:.2f}"])

# Print the table
print(table)

# Load all models into memory and get total memory usage
print("\n-------------------------")
print("Loading all models into memory")
all_models = []
total_memory_before = get_memory_usage()
for model_name in sd_models[1:]:
    all_models.append(load_model(f"D:\\miniconda\\UNSW-NB15\\ngfw\\models\\{model_name}_model_ANN5_hp.keras"))
gc.collect()  # Perform garbage collection

total_memory_after_gc = get_memory_usage()

print(f"Memory usage after loading all models: {total_memory_after_gc:.2f} MB")
print(f"Memory usage after unloading models: {total_memory_after_gc:.2f} MB")
