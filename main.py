from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev
import pandas as pd
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("AnomalyDetection").getOrCreate() # Створення Spark-сесії

file_path = "transactions.csv"  #  Завантаження даних
df = spark.read.csv(file_path, header=True, inferSchema=True)

df = df.dropna()  # Очищення та обробка даних

stats = df.select(mean(col("amount")).alias("mean"), stddev(col("amount")).alias("stddev")).collect()  #  Обчислення статистичних метрик
mean_value = stats[0]["mean"]
stddev_value = stats[0]["stddev"]

anomaly_threshold = mean_value + 3 * stddev_value  # Виявлення аномальних транзакцій (більше 3 стандартних відхилень)
anomalies = df.filter(col("amount") > anomaly_threshold)

print("Аномальні транзакції:")  # Виведення результатів
anomalies.show()

anomalies.write.csv("anomalies.csv", header=True, mode="overwrite") # Збереження аномалій

pandas_df = df.select("amount").toPandas()  #  Візуалізація
pandas_anomalies = anomalies.select("amount").toPandas()

plt.figure(figsize=(10, 6))
plt.hist(pandas_df["amount"], bins=50, alpha=0.5, label="Звичайні транзакції")
plt.hist(pandas_anomalies["amount"], bins=50, alpha=0.9, color='red', label="Аномалії")
plt.axvline(anomaly_threshold, color='black', linestyle='dashed', linewidth=2, label="Поріг аномалій")
plt.xlabel("Сума транзакції")
plt.ylabel("Кількість")
plt.title("Розподіл транзакцій та аномалії")
plt.legend()
plt.show()

spark.stop()  # Завершення сесії
