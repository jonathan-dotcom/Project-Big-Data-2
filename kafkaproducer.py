import time
import random
import pandas as pd
from kafka import KafkaProducer

# Configuration
KAFKA_TOPIC = "online_retail_topic"
KAFKA_BROKER = "localhost:9092"  # Update with your Kafka broker address
DATA_FILE = "./online_retail_II.xlsx"  # Path to the uploaded Excel dataset

# Initialize Kafka Producer
producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER)

# Read dataset
data = pd.read_excel(DATA_FILE)
data['InvoiceDate'] = data['InvoiceDate'].astype(str)

# Send each row to Kafka topic with random delay
for index, row in data.iterrows():
    message = row.to_json()
    print(f"Sending message to Kafka: {message}")  # Print message to terminal
    producer.send(KAFKA_TOPIC, message.encode('utf-8'))
    time.sleep(random.uniform(0.1, 1))  # Random delay between 0.1 to 1 seconds

producer.flush()
producer.close()
