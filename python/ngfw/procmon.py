import psutil
import os
import time

# Get the current process ID
pid = os.getpid()

# Get the process object
proc = psutil.Process(pid)

# Get memory usage, CPU usage, etc.
while True:
    print(f"Memory Usage: {proc.memory_info().rss / 1024 ** 2} MB")
    print(f"CPU Usage: {proc.cpu_percent(interval=1)}%")
    #print(f"Open Files: {proc.open_files()}")
    print(f"Connections: {proc.connections()}")
    time.sleep(1)  # Pause for a second before next update
