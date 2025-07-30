import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_data
def load_quartiers(filepath):
    df = pd.read_csv(filepath)
    required_cols = {"id_quartier", "Nom_Fokontany"}
    if not required_cols.issubset(df.columns):
        st.error(f"Le fichier doit contenir les colonnes : {', '.join(required_cols)}")
        st.stop()
    return df

quartiers_df = load_quartiers("fokontany_tana_avec_id.csv")

@st.cache_resource
def load_model():
    return joblib.load("modele_ridge_pipeline.joblib")

model = load_model()

st.title("Prédiction du loyer mensuel à Antananarivo")

quartier_nom = st.selectbox("Quartier", quartiers_df["Nom_Fokontany"].tolist())

superficie = st.number_input("Superficie (m²)", min_value=10.0, max_value=500.0, value=50.0, step=1.0)
surface_terrain = st.number_input("Surface terrain (m²)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
nombre_chambres = st.number_input("Nombre de chambres", min_value=0, max_value=10, value=2, step=1)
nombre_salles_eau = st.number_input("Nombre de salles d'eau", min_value=0, max_value=10, value=1, step=1)

meuble = st.checkbox("Meublé ?")


douche_wc = st.selectbox("Douche/WC", options=["interieur", "exterieur"])
type_d_acces = st.selectbox("Type d'accès", options=["sans", "moto", "voiture", "voiture_avec_parking"])
etat_general = st.selectbox("État général", options=["bon", "moyen", "mauvais"])
type_logement = st.selectbox("Type de logement", options=["appartement", "maison", "studio", "villa"]) 
input_data = pd.DataFrame({
    "quartier": [quartier_nom],
    "superficie": [superficie],
    "surface_terrain": [surface_terrain],
    "nombre_chambres": [nombre_chambres],
    "nombre_salles_eau": [nombre_salles_eau],
    "meuble": [int(meuble)], 
    "douche_wc": [douche_wc],
    "type_d_acces": [type_d_acces],
    "etat_general": [etat_general],
    "type_logement": [type_logement]
})

if st.button("🔮 Estimer le loyer"):
    input_encoded = pd.get_dummies(input_data)
    model_features = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    if prediction < 0:
        st.warning(" Le modèle prédit un loyer négatif, veuillez vérifier les entrées.")
    else:
        st.success(f" Loyer mensuel prédit : {int(prediction):,} MGA")
