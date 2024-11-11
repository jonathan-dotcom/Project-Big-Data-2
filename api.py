import os

from flask import Flask, request, jsonify
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
from pyspark.sql import SparkSession
app = Flask(__name__)
os.environ['PYSPARK_PYTHON'] = r'C:\Users\jonat\anaconda3\envs\pyspark_env\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\jonat\anaconda3\envs\pyspark_env\python.exe'
# Inisialisasi SparkSession dengan konfigurasi python executable
spark = SparkSession.builder \
    .getOrCreate()

# Direktori model
model_directory = "./models"

# Memuat model
model1 = KMeansModel.load(os.path.join(model_directory, "model1"))
model2 = KMeansModel.load(os.path.join(model_directory, "model2"))
model3 = KMeansModel.load(os.path.join(model_directory, "model3"))

def predict_cluster(data, model):
    # Membuat DataFrame dari input data
    df = spark.createDataFrame([Row(**data)])
    assembler = VectorAssembler(inputCols=["TotalQuantity", "TotalAmount", "NumInvoices"], outputCol="features")
    df = assembler.transform(df)
    prediction = model.transform(df).collect()[0]['prediction']
    return prediction

# Endpoint untuk Model 1
@app.route('/predict_cluster_model1', methods=['POST'])
def predict_cluster_model1():
    data = request.get_json()
    prediction = predict_cluster(data, model1)
    return jsonify({'cluster': int(prediction)})

# Endpoint untuk Model 2
@app.route('/predict_cluster_model2', methods=['POST'])
def predict_cluster_model2():
    data = request.get_json()
    prediction = predict_cluster(data, model2)
    return jsonify({'cluster': int(prediction)})

# Endpoint untuk Model 3
@app.route('/predict_cluster_model3', methods=['POST'])
def predict_cluster_model3():
    data = request.get_json()
    prediction = predict_cluster(data, model3)
    return jsonify({'cluster': int(prediction)})

if __name__ == '__main__':
    app.run(port=5000)
