import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ["unix_time", "model", "rec_no", "speed", "testing_time"]  # Modify based on structure
df = pd.read_csv("log_m1_2.csv", names=column_names, header=None)

# Display first few rows
print(df.head())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Plot Distribution of Single Record Testing Time
plt.figure(figsize=(10, 5))
sns.histplot(df['testing_time'], bins=30, kde=True)
plt.xlabel("Testing Time")
plt.ylabel("Frequency")
plt.title("Distribution of Single Record Testing Time")
plt.savefig("multi_testing_time_distribution.png")  # Save figure
plt.close()

# Boxplot to Identify Outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['testing_time'])
plt.xlabel("Testing Time")
plt.title("Boxplot of Single Record Testing Time")
plt.savefig("multi_testing_time_boxplot.png")  # Save figure
plt.close()

print("Figures saved as 'smulti_testing_time_distribution.png' and 'multi_testing_time_boxplot.png'")
