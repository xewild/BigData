from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Spark Session
spark = SparkSession.builder \
    .appName("TSA Detection - ML Inference") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Charger les artefacts ML
model   = joblib.load("model.pkl")
scaler  = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

print("Modèle chargé avec succès !")
print(f"Nombre de features attendues : {len(columns)}")


# Fonction de prédiction
def detect_batch(data_list):

    try:
        # 1 Convertir en DataFrame pandas
        df = pd.DataFrame(data_list)

        # 2 Supprimer la colonne identifiant si présente
        cols_to_drop = ["CASE_NO_PATIENT'S", "Childhood Autism Rating Scale", "Qchat-10_Score"]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

        # 3  Supprimer la cible si présente
        df.drop(columns=["ASD_traits"], inplace=True, errors="ignore")

        # 4  Normaliser la casse Ethnicity
        if "Ethnicity" in df.columns:
            df["Ethnicity"] = df["Ethnicity"].str.strip().str.title()

        # 5  Remplacer les valeurs manquantes
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)

        cat_cols = df.select_dtypes(include=["object"]).columns
        df[cat_cols] = df[cat_cols].fillna("No")

        # 6  Encodage one-hot
        X = pd.get_dummies(df)

        # 7  Aligner avec les colonnes d'entraînement
        X = X.reindex(columns=columns, fill_value=0)

        # 8  Normalisation
        X_scaled = scaler.transform(X)

        # 9  Prédiction
        proba   = model.predict_proba(X_scaled)
        prob_no  = proba[:, 0]
        prob_yes = proba[:, 1]
        pred     = (prob_yes >= 0.5).astype(int)

        # 10  Niveau de risque
        risk_level = np.where(
            prob_yes >= 0.7, "HIGH",
            np.where(prob_yes >= 0.4, "MEDIUM", "LOW")
        )

        # 11  Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 12  Construire le DataFrame résultat
        result_df = pd.DataFrame({
            "detection":          pred,
            "probability_no_tsa": prob_no,
            "probability_tsa":    prob_yes,
            "risk_level":         risk_level,
            "timestamp":          timestamp
        })

        # 13 Sauvegarder dans HDFS via Spark
        try:
            spark_df = spark.createDataFrame(result_df)
            spark_df.write \
                .mode("append") \
                .format("parquet") \
                .option("path", "hdfs://localhost:9000/user/tsa/results") \
                .save()
            print("Résultat sauvegardé dans HDFS")

        except Exception as e:
            print("HDFS/Spark non disponible, sauvegarde ignorée :", e)

        return result_df

    except Exception as e:
        print("Erreur lors de la prédiction :", e)
        return pd.DataFrame({
            "detection":          [0],
            "probability_tsa":    [0.0],
            "probability_no_tsa": [1.0],
            "risk_level":         ["ERROR"],
            "timestamp":          [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
