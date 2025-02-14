from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev
import pandas as pd
import matplotlib.pyplot as plt

# Create a Spark session
spark = SparkSession.builder.appName("AnomalyDetection").getOrCreate()

# Load data from a CSV file
file_path = "transactions.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Clean the data by removing rows with missing values
df = df.dropna()

# Calculate statistical metrics (mean and standard deviation) for the "amount" column
stats = df.select(mean(col("amount")).alias("mean"), stddev(col("amount")).alias("stddev")).collect()
mean_value = stats[0]["mean"]
stddev_value = stats[0]["stddev"]

# Define the anomaly threshold (3 standard deviations above the mean)
anomaly_threshold = mean_value + 3 * stddev_value

# Detect anomalies (transactions exceeding the threshold)
anomalies = df.filter(col("amount") > anomaly_threshold)

# Display the anomalies
print("Anomalous Transactions:")
anomalies.show()

# Save the anomalies to a CSV file
anomalies.write.csv("anomalies.csv", header=True, mode="overwrite")

# Convert Spark DataFrames to Pandas DataFrames for visualization
pandas_df = df.select("amount").toPandas()
pandas_anomalies = anomalies.select("amount").toPandas()

# Visualize the distribution of transactions and anomalies
plt.figure(figsize=(10, 6))
plt.hist(pandas_df["amount"], bins=50, alpha=0.5, label="Normal Transactions")
plt.hist(pandas_anomalies["amount"], bins=50, alpha=0.9, color='red', label="Anomalies")
plt.axvline(anomaly_threshold, color='black', linestyle='dashed', linewidth=2, label="Anomaly Threshold")
plt.xlabel("Transaction Amount")
plt.ylabel("Count")
plt.title("Transaction Distribution and Anomalies")
plt.legend()
plt.show()

# Stop the Spark session
spark.stop()
