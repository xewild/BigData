import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Charger le dataset 
df = pd.read_csv("data_csv.csv")

print("Aperçu dataset :")
print(df.head())
print("\nShape initial :", df.shape)


# Nettoyage

# Supprimer la colonne identifiant patient (non pertinente)
cols_to_drop = ["CASE_NO_PATIENT'S", "Childhood Autism Rating Scale", "Qchat-10_Score"]

for col in cols_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Remplacer les valeurs manquantes numériques par la médiane
df["Social_Responsiveness_Scale"] = df["Social_Responsiveness_Scale"].fillna(
    df["Social_Responsiveness_Scale"].median()
)

# Remplacer les valeurs manquantes catégorielles par le mode
for col in ["Depression", "Social/Behavioural Issues"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Normaliser la casse de l'ethnicité (ex: 'middle eastern' → 'Middle Eastern')
df["Ethnicity"] = df["Ethnicity"].str.strip().str.title()

print("\nShape après nettoyage :", df.shape)
print("Valeurs manquantes restantes :", df.isnull().sum().sum())


# Séparer X / y
if "ASD_traits" not in df.columns:
    raise ValueError("Colonne cible 'ASD_traits' introuvable dans le dataset")

y = df["ASD_traits"].str.strip()
X = df.drop("ASD_traits", axis=1)


# Encodage des variables catégorielles 
X = pd.get_dummies(X)

# Sauvegarder la structure des colonnes pour l'inférence
joblib.dump(X.columns.tolist(), "columns.pkl")
print("\nNombre de features après encodage :", len(X.columns))

# Encoder la cible
y = y.map({"No": 0, "Yes": 1})

print("Distribution de la cible :")
print(y.value_counts())


# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain : {X_train.shape[0]} instances")
print(f"Test  : {X_test.shape[0]} instances")


# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# Modèle Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"   # important car classes légèrement déséquilibrées
)

print("\nEntraînement du modèle...")
model.fit(X_train, y_train)


# Évaluation
y_pred = model.predict(X_test)

print("RÉSULTATS DU MODÈLE")
print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=["No TSA", "TSA"]))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))


# Sauvegarde
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nFichiers générés :")
print("  model.pkl   — modèle Random Forest entraîné")
print("  scaler.pkl  — StandardScaler ajusté")
print("  columns.pkl — structure des colonnes du dataset")
print("\nEntraînement terminé avec succès !")
