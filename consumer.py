from kafka import KafkaConsumer
import json
import requests
from spark_model import detect_batch

# Consumer Kafka
consumer = KafkaConsumer(
    'tsa-topic',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("En attente des données Kafka...")

for message in consumer:
    data = message.value
    print("\nDonnée reçue :", data)

    try:
        # Prédiction ML
        result = detect_batch([data])
        row    = result.iloc[0]

        # Formatage du résultat 
        result_data = {
            "detection":          int(row["detection"]),
            "probability_tsa":    round(float(row["probability_tsa"])    * 100, 2),
            "probability_no_tsa": round(float(row["probability_no_tsa"]) * 100, 2),
            "risk_level":         str(row["risk_level"]),
            "timestamp":          str(row["timestamp"])
        }

        print("\nRésultat :", result_data)

        # Envoi au producer via HTTP → WebSocket
        r = requests.post(
            "http://localhost:5000/emit-result",
            json=result_data,
            timeout=5
        )
        print("Emit status :", r.status_code)

    except Exception as e:
        print("Erreur traitement :", str(e))
