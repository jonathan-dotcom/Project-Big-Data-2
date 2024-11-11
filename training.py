from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.recommendation import ALS
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("KafkaSparkModelTraining") \
    .getOrCreate()

# Directory containing the batch datasets
batch_data_dir = "./data"
model_save_dir = "./models"

# Function to train KMeans, ALS, and Decision Tree models
def train_models(data, model_save_dir, model_name_prefix):
    assembler = VectorAssembler(inputCols=["Quantity", "Price"], outputCol="features")
    data = assembler.transform(data)

    # Train KMeans model
    kmeans = KMeans().setK(3).setSeed(1)
    kmeans_model = kmeans.fit(data)
    kmeans_model.write().overwrite().save(f"{model_save_dir}/{model_name_prefix}_kmeans.model")
    print(f"Saved KMeans model: {model_name_prefix}_kmeans.model")

    # Prepare data for ALS model
    data = data.withColumnRenamed("Customer ID", "CustomerID")
    data = data.withColumn("CustomerID", data["CustomerID"].cast("integer"))
    data = data.withColumn("StockCode", data["StockCode"].cast("integer"))  # Cast StockCode to integer

    # Filter out rows with null values in CustomerID or StockCode
    data_filtered = data.dropna(subset=["CustomerID", "StockCode"])

    # Train ALS recommendation model
    als = ALS(userCol="CustomerID", itemCol="StockCode", ratingCol="Quantity", implicitPrefs=True, coldStartStrategy="drop")
    als_model = als.fit(data_filtered)
    als_model.write().overwrite().save(f"{model_save_dir}/{model_name_prefix}_als.model")
    print(f"Saved ALS model: {model_name_prefix}_als.model")

    # Train Decision Tree classifier
    data = data.withColumn("label", (data["Quantity"] > 0).cast("int"))
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    dt_model = dt.fit(data)
    dt_model.write().overwrite().save(f"{model_save_dir}/{model_name_prefix}_dt.model")
    print(f"Saved Decision Tree model: {model_name_prefix}_dt.model")

# Load and train on progressively larger batches
batch_files = sorted(os.listdir(batch_data_dir))

data_accumulated = None
for i, batch_file in enumerate(batch_files, 1):
    batch_path = os.path.join(batch_data_dir, batch_file)
    data = spark.read.json(batch_path)

    data_accumulated = data if data_accumulated is None else data_accumulated.union(data)
    train_models(data_accumulated, model_save_dir, f"progressive_model_{i}")

spark.stop()
