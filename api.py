from flask import Flask, request, jsonify
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import os
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ModelServingAPI") \
    .getOrCreate()

# Load models
model_dir = "./models"
models = {
    "kmeans": [KMeansModel.load(os.path.join(model_dir, f"progressive_model_{i}_kmeans.model")) for i in range(1, 4)],
    "als": [ALSModel.load(os.path.join(model_dir, f"progressive_model_{i}_als.model")) for i in range(1, 4)],
    "dt": [DecisionTreeClassificationModel.load(os.path.join(model_dir, f"progressive_model_{i}_dt.model")) for i in
           range(1, 4)]
}


@app.route('/predict/<model_type>/<model_number>', methods=['POST'])
def predict(model_type, model_number):
    model_number = int(model_number) - 1

    if model_type not in models or model_number not in range(3):
        return jsonify({"error": "Invalid model type or number"}), 400

    model = models[model_type][model_number]
    input_data = request.json
    quantity = input_data.get("quantity", 0)
    unit_price = input_data.get("unit_price", 0)

    print(f"Received prediction request for {model_type} model {model_number + 1}")  # Print request info

    # Convert to DataFrame for prediction
    df = spark.createDataFrame([(quantity, unit_price)], ["Quantity", "UnitPrice"])
    assembler = VectorAssembler(inputCols=["Quantity", "UnitPrice"], outputCol="features")
    data = assembler.transform(df)

    if model_type == "kmeans":
        prediction = model.transform(data).select("prediction").collect()[0]["prediction"]
    elif model_type == "als":
        customer_id = input_data.get("customer_id", 0)
        data = spark.createDataFrame([(customer_id, quantity)], ["CustomerID", "StockCode"])
        recommendations = model.recommendForUserSubset(data, 5).collect()
        prediction = recommendations
    elif model_type == "dt":
        prediction = model.transform(data).select("prediction").collect()[0]["prediction"]

    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
