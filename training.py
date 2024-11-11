from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import os

# Inisialisasi SparkSession
spark = SparkSession.builder.appName("CustomerSegmentation") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

# Direktori data dan model
data_directory = "./data"
model_directory = "./models"

# Membuat direktori model jika belum ada
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Mendapatkan daftar file batch dan mengurutkannya
batch_files = sorted(os.listdir(data_directory))
total_batches = len(batch_files)

# Inisialisasi variabel
accumulated_df = None
model1_trained = False
model2_trained = False
model3_trained = False

# Menentukan threshold batch untuk setiap model
model1_threshold = total_batches // 3
model2_threshold = (total_batches * 2) // 3
model3_threshold = total_batches


def train_model(dataframe, model_name):
    # Menghapus data yang tidak memiliki Customer ID
    df = dataframe.na.drop(subset=["Customer ID"])

    # Konversi tipe data
    df = df.withColumn("Quantity", col("Quantity").cast("float"))
    df = df.withColumn("Price", col("Price").cast("float"))
    df = df.withColumn("InvoiceDate", to_timestamp("InvoiceDate", "yyyy-MM-dd HH:mm:ss"))

    # Menghitung TotalAmount
    df = df.withColumn("TotalAmount", col("Quantity") * col("Price"))

    # Menghitung fitur per Customer ID
    customer_df = df.groupBy("Customer ID").agg(
        sum("Quantity").alias("TotalQuantity"),
        sum("TotalAmount").alias("TotalAmount"),
        countDistinct("Invoice").alias("NumInvoices")
    )

    # Membuat fitur vektor
    assembler = VectorAssembler(inputCols=["TotalQuantity", "TotalAmount", "NumInvoices"], outputCol="features")
    feature_df = assembler.transform(customer_df)

    # Training model KMeans
    kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=4, seed=1)
    model = kmeans.fit(feature_df)

    # Menyimpan model
    model.save(os.path.join(model_directory, model_name))
    print(f"{model_name} telah dilatih dan disimpan.")


# Membaca batch data dan melakukan training model sesuai threshold
for idx, batch_file in enumerate(batch_files):
    batch_path = os.path.join(data_directory, batch_file)
    batch_df = spark.read.json(batch_path)

    if accumulated_df is None:
        accumulated_df = batch_df
    else:
        accumulated_df = accumulated_df.union(batch_df)

    # Training Model 1
    if not model1_trained and idx + 1 >= model1_threshold:
        train_model(accumulated_df, "model1")
        model1_trained = True

    # Training Model 2
    if not model2_trained and idx + 1 >= model2_threshold:
        train_model(accumulated_df, "model2")
        model2_trained = True

    # Training Model 3
    if not model3_trained and idx + 1 >= model3_threshold:
        train_model(accumulated_df, "model3")
        model3_trained = True

# Menghentikan SparkSession
spark.stop()
