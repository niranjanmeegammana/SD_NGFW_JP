import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define column names
columns = ["count", "model", "speed", "loading_time"]

# Load data into a DataFrame
data = pd.read_csv("log_m2_1.csv", names=columns)

# Display the first few rows
print(data.head())
print("--------------------")
print(data.describe())
print("--------------------")
model_group = data.groupby("model").agg({
    "speed": ["mean", "min", "max"],
    "loading_time": ["mean", "min", "max"]
})
print(model_group)

# Set Seaborn style
sns.set_style("whitegrid")

# Save Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="model", y="loading_time", data=data)
plt.title("Loading Time Distribution by Model")
plt.xticks(rotation=45)
plt.savefig("multi_loading_time_distribution.png", dpi=300, bbox_inches="tight")  # Save as PNG
plt.show()

# Save Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x="speed", y="loading_time", hue="model", data=data)
plt.title("Speed vs Loading Time")
plt.savefig("multi_speed_vs_loading_time.png", dpi=300, bbox_inches="tight")  # Save as PNG
plt.show()

# Compute the fastest models
fastest_model = data.groupby("model")["loading_time"].mean().idxmin()
print("Fastest Model (Lowest Avg Loading Time):", fastest_model)

fastest_speed_model = data.groupby("model")["speed"].mean().idxmax()
print("Model with Highest Avg Speed:", fastest_speed_model)

# Print correlation matrix
print(data.corr())

# Compare deep vs shallow models
deep_models = data[data["model"].str.contains("deep")]
shallow_models = data[data["model"].str.contains("shallow")]

print("Deep Models Avg Loading Time:", deep_models["loading_time"].mean())
print("Shallow Models Avg Loading Time:", shallow_models["loading_time"].mean())
