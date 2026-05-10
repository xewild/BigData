from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
from kafka import KafkaProducer
import json

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


# HOME
@app.route('/')
def home():
    return render_template("index.html")


# SEND DATA
@app.route('/send', methods=['POST'])
def send_data():
    try:
        data = request.json
        print("Données reçues :", data)

        # NORMALISATION selon les colonnes du nouveau dataset
        formatted_data = {

            # 10 scores AQ
            "A1": int(data.get("A1", 0)),
            "A2": int(data.get("A2", 0)),
            "A3": int(data.get("A3", 0)),
            "A4": int(data.get("A4", 0)),
            "A5": int(data.get("A5", 0)),
            "A6": int(data.get("A6", 0)),
            "A7": int(data.get("A7", 0)),
            "A8": int(data.get("A8", 0)),
            "A9": int(data.get("A9", 0)),
            "A10_Autism_Spectrum_Quotient": int(data.get("A10_Autism_Spectrum_Quotient", 0)),

            # Scores cliniques
            "Social_Responsiveness_Scale":   int(data.get("Social_Responsiveness_Scale", 0)),
            "Age_Years":                     int(data.get("Age_Years", 0)),

            # Variables binaires cliniques
            "Speech Delay/Language Disorder":                    data.get("Speech_Delay", "No"),
            "Learning disorder":                                 data.get("Learning_disorder", "No"),
            "Genetic_Disorders":                                 data.get("Genetic_Disorders", "No"),
            "Depression":                                        data.get("Depression", "No"),
            "Global developmental delay/intellectual disability": data.get("Global_delay", "No"),
            "Social/Behavioural Issues":                         data.get("Social_Issues", "No"),
            "Anxiety_disorder":                                  data.get("Anxiety_disorder", "No"),

            # Infos démographiques
            "Sex":                    data.get("Sex", "M"),
            "Ethnicity":              data.get("Ethnicity", "Others"),
            "Jaundice":               data.get("Jaundice", "No"),
            "Family_mem_with_ASD":    data.get("Family_mem_with_ASD", "No"),
            "Who_completed_the_test": data.get("Who_completed_the_test", "Self"),
        }

        # ENVOI KAFKA
        producer.send('tsa-topic', formatted_data)
        producer.flush()
        print("Données envoyées à Kafka")

        return jsonify({"message": "Analyse en cours..."})

    except Exception as e:
        print("Erreur :", str(e))
        return jsonify({"error": str(e)}), 500


# EMIT RESULT — appelé par consumer.py 
@app.route('/emit-result', methods=['POST'])
def emit_result():
    try:
        data = request.json
        print("Résultat reçu du consumer :", data)
        socketio.emit('result', data)
        return jsonify({"ok": True})

    except Exception as e:
        print("Erreur emit :", str(e))
        return jsonify({"error": str(e)}), 500


# RUN 
if __name__ == '__main__':
    socketio.run(app, debug=True)
