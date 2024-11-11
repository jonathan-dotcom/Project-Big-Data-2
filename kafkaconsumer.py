from kafka import KafkaConsumer
import json
import os

# Configuration
KAFKA_TOPIC = "online_retail_topic"
KAFKA_BROKER = "localhost:9092"  # Update with your Kafka broker address
BATCH_SIZE = 1000  # Number of messages per batch
OUTPUT_DIR = "./data"  # Directory to save batches

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="consumer-group"
)

batch = []
batch_count = 1

for message in consumer:
    batch.append(json.loads(message.value.decode('utf-8')))

    if len(batch) >= BATCH_SIZE:
        output_file = os.path.join(OUTPUT_DIR, f"batch_{batch_count}.json")
        with open(output_file, 'w') as f:
            json.dump(batch, f)
        print(f"Saved batch {batch_count} with {BATCH_SIZE} messages")  # Print batch-saving progress
        batch = []
        batch_count += 1
