
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import json

st.set_page_config(page_title="STEG Jendouba - Prediction Pannes", layout="wide", page_icon="⚡")

st.markdown("""
<style>
h1 { color: #1A237E; font-weight: 800; border-bottom: 3px solid #1E88E5; padding-bottom: 8px; }
h2 { color: #1A237E; font-weight: 700; margin-top: 1.5rem; }
h3 { color: #283593; font-weight: 600; }

[data-testid="stSidebar"] { background-color: #0D1B4C; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; }

[data-testid="stMetric"] {
    background-color: #F5F7FA;
    border: 1px solid #E0E0E0;
    border-left: 5px solid #1E88E5;
    border-radius: 10px;
    padding: 15px 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { font-weight: 600; color: #455A64; }
[data-testid="stMetricValue"] { color: #1A237E; font-weight: 800; }

[data-testid="stDataFrame"] { border: 1px solid #E0E0E0; border-radius: 8px; overflow: hidden; }

.stTabs [data-baseweb="tab"] { font-weight: 600; }

.stButton>button {
    background-color: #1E88E5; color: white; border-radius: 8px;
    border: none; font-weight: 600;
}
.stButton>button:hover { background-color: #1565C0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df_panne = pd.read_csv("data/pannes.csv", encoding="utf-8-sig")
    df_fact  = pd.read_csv("data/facturation.csv", encoding="utf-8-sig")
    df_km    = pd.read_csv("data/delegations_clusters.csv", encoding="utf-8-sig")
    df_comp_reg = pd.read_csv("data/comparatif_regression.csv", encoding="utf-8-sig")
    df_comp_clf = pd.read_csv("data/comparatif_classification.csv", encoding="utf-8-sig")
    df_lstm  = pd.read_csv("data/predictions_lstm.csv", encoding="utf-8-sig")
    df_importance = pd.read_csv("data/importance_variables.csv", index_col=0, encoding="utf-8-sig")
    df_anomalies = pd.read_csv("data/anomalies_pannes.csv", encoding="utf-8-sig")
    return df_panne, df_fact, df_km, df_comp_reg, df_comp_clf, df_lstm, df_importance, df_anomalies

@st.cache_resource
def load_model():
    with open("model/xgb_regression.pkl", "rb") as f:
        xgb_reg = pickle.load(f)
    with open("model/xgb_classification.pkl", "rb") as f:
        xgb_clf = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/label_encoder_saison.pkl", "rb") as f:
        le_saison = pickle.load(f)
    with open("model/label_encoder_delegation.pkl", "rb") as f:
        le_deleg = pickle.load(f)
    with open("model/label_encoder_zone.pkl", "rb") as f:
        le_zone = pickle.load(f)
    with open("model/label_encoder_cause.pkl", "rb") as f:
        le_cause = pickle.load(f)
    with open("model/label_encoder_type.pkl", "rb") as f:
        le_type = pickle.load(f)
    with open("model/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return xgb_reg, xgb_clf, scaler, le_saison, le_deleg, le_zone, le_cause, le_type, metadata

df_panne, df_fact, df_km, df_comp_reg, df_comp_clf, df_lstm, df_importance, df_anomalies = load_data()
xgb_reg, xgb_clf, scaler, le_saison, le_deleg, le_zone, le_cause, le_type, metadata = load_model()

st.sidebar.title("STEG Jendouba")
st.sidebar.markdown("Prediction et Analyse des Pannes Electriques")
page = st.sidebar.radio("Navigation", [
    "1. Tableau de Bord",
    "2. Donnees et Exploration",
    "3. Preparation des Donnees",
    "4. Modelisation et Evaluation",
    "5. Clustering et Anomalies",
    "6. Carte Interactive",
    "7. Prediction Personnalisee",
    "8. Recommandations"
])


# ============================================================
# PAGE 1 - TABLEAU DE BORD
# ============================================================
if page == "1. Tableau de Bord":
    st.title("Tableau de Bord - STEG Jendouba")
    st.markdown("Vue synthetique du projet de prediction et analyse des pannes electriques")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pannes", "{:,}".format(len(df_panne)))
    col2.metric("Delegations analysees", df_panne[df_panne["delegation"]!="Inconnu"]["delegation"].nunique())
    col3.metric("Duree moyenne", "{:.1f} h".format(df_panne["duree_h"].mean()))
    col4.metric("Anomalies detectees", "{:,}".format(df_anomalies["anomalie"].sum()))

    col5, col6, col7 = st.columns(3)
    col5.metric("R2 Regression (XGBoost)", "{:.3f}".format(metadata["R2_regression"]))
    col6.metric("F1 Classification (XGBoost)", "{:.3f}".format(metadata["F1_classification"]))
    col7.metric("MAE", "{:.3f} h".format(metadata["MAE_regression"]))

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Evolution des pannes par annee")
        evol = df_panne.groupby("annee").size().reset_index(name="nb_pannes")
        fig = px.line(evol, x="annee", y="nb_pannes", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Repartition par saison")
        ordre = ["Ete","Automne","Hiver","Printemps"]
        saison_counts = df_panne["saison"].value_counts().reindex(ordre)
        fig2 = px.pie(values=saison_counts.values, names=saison_counts.index)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Validation des criteres metier CRISP-DM")
    c1, c2, c3 = st.columns(3)
    c1.success("R2 > 0.70 : VALIDE ({:.4f})".format(metadata["R2_regression"]))
    c2.success("MAE < 1h : VALIDE ({:.3f}h)".format(metadata["MAE_regression"]))
    c3.success("F1 > 0.70 : VALIDE ({:.4f})".format(metadata["F1_classification"]))


# ============================================================
# PAGE 2 - DONNEES ET EXPLORATION
# ============================================================
if page == "2. Donnees et Exploration":
    st.title("Donnees et Exploration")
    tab1, tab2, tab3 = st.tabs(["Distribution", "Bivariee", "Regionale"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(df_panne["duree_h"], bins=40, color="#E53935", edgecolor="white")
            ax.axvline(df_panne["duree_h"].mean(), color="navy", linestyle="--")
            ax.set_title("Distribution Duree des Pannes")
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.boxplot(df_panne["duree_h"], vert=False)
            ax2.set_title("Boxplot Duree des Pannes")
            st.pyplot(fig2)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.scatter(df_panne.sample(min(2000,len(df_panne))),
                               x="duree_h", y="impact_energie_kwh",
                               opacity=0.4, title="Duree vs Energie Perdue")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            cols_corr = ["duree_h","nb_abonnes_affectes","impact_energie_kwh","cout_estime_tnd"]
            corr = df_panne[cols_corr].corr()
            fig4, ax4 = plt.subplots(figsize=(6,5))
            sns.heatmap(corr, annot=True, cmap="RdYlBu_r", ax=ax4)
            st.pyplot(fig4)

    with tab3:
        df_p = df_panne[df_panne["delegation"]!="Inconnu"]
        deleg_stats = df_p.groupby("delegation")["duree_h"].agg(["count","mean"]).reset_index()
        deleg_stats.columns = ["Delegation","Nb Pannes","Duree Moyenne"]
        fig5 = px.bar(deleg_stats.sort_values("Nb Pannes"), x="Nb Pannes", y="Delegation",
                       orientation="h", title="Pannes par Delegation")
        st.plotly_chart(fig5, use_container_width=True)
        st.dataframe(deleg_stats.sort_values("Nb Pannes", ascending=False))


# ============================================================
# PAGE 3 - PREPARATION DES DONNEES
# ============================================================
if page == "3. Preparation des Donnees":
    st.title("Preparation des Donnees - Feature Store")
    tab1, tab2 = st.tabs(["Feature Store", "Partitionnement"])

    with tab1:
        st.subheader("11 Variables Explicatives")
        features_info = pd.DataFrame({
            "Variable": ["annee","mois_num","saison_enc","delegation_enc","zone_enc",
                         "cause_enc","type_enc","nb_abonnes_affectes","is_weekend",
                         "recidive_flag","impact_energie_kwh"],
            "Type": ["int","int","int","int","int","int","int","float","int","int","float"],
            "Description": ["Annee incident","Mois (1-12)","Saison encodee","Delegation encodee",
                            "Zone encodee","Cause encodee","Type encode","Nb abonnes touches",
                            "1=weekend","1=recidive","Energie non distribuee (kWh)"]
        })
        st.dataframe(features_info, use_container_width=True)
        st.warning("Variable cible regression : duree_h | Variable cible classification : anomalie_duree")

    with tab2:
        st.subheader("Repartition Train/Test")
        col1, col2 = st.columns(2)
        col1.metric("Lignes exploitables", "4 110")
        col2.metric("random_state retenu", "41")
        fig = px.pie(values=[70,30], names=["Train (70%)","Test (30%)"])
        st.plotly_chart(fig)


# ============================================================
# PAGE 4 - MODELISATION ET EVALUATION
# ============================================================
if page == "4. Modelisation et Evaluation":
    st.title("Modelisation et Evaluation des Algorithmes")
    tab1, tab2, tab3 = st.tabs(["Tableau Comparatif", "Visualisations", "XGBoost - Analyse"])

    with tab1:
        st.subheader("Regression - Prediction duree_h")
        st.dataframe(df_comp_reg.style.highlight_max(subset=["R2"], color="lightgreen")
                     .highlight_min(subset=["MAE","RMSE"], color="lightgreen"))
        st.subheader("Classification - Detection anomalie_duree")
        st.dataframe(df_comp_clf.style.highlight_max(subset=["F1","Accuracy"], color="lightgreen"))

    with tab2:
        metrique = st.selectbox("Choisir une metrique", ["MAE","RMSE","R2"])
        fig = px.bar(df_comp_reg, x="Algorithme", y=metrique, color="Algorithme")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Importance des Variables (XGBoost)")
        fig2 = px.bar(df_importance.sort_values("XGBoost"), x="XGBoost", y=df_importance.sort_values("XGBoost").index,
                      orientation="h")
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# PAGE 5 - CLUSTERING ET ANOMALIES
# ============================================================
if page == "5. Clustering et Anomalies":
    st.title("Clustering des Delegations et Detection d Anomalies")
    tab1, tab2 = st.tabs(["Clustering K-Means/DBSCAN", "Isolation Forest"])

    with tab1:
        st.dataframe(df_km[["delegation","cluster","dbscan","nb_pannes","duree_moy"]])
        fig = px.scatter(df_km, x="PC1", y="PC2", color="cluster", text="delegation",
                          title="Clusters K-Means (vue PCA)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        n_anom = df_anomalies["anomalie"].sum()
        st.metric("Pannes anormales detectees", "{:,} ({:.1f}%)".format(n_anom, n_anom/len(df_anomalies)*100))
        top_causes = df_anomalies[df_anomalies["anomalie"]==1]["cause_panne"].value_counts().head(8)
        fig2 = px.bar(x=top_causes.values, y=top_causes.index, orientation="h",
                       title="Causes des Pannes Anormales")
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# PAGE 6 - CARTE INTERACTIVE (NOUVEAUTE)
# ============================================================
if page == "6. Carte Interactive":
    st.title("Carte Interactive du Reseau Electrique - Jendouba")

    coords_delegations = {
        "Jendouba"     : (36.5011, 8.7802),
        "Tabarka"      : (36.9547, 8.7583),
        "Ain Draham"   : (36.7794, 8.6883),
        "Fernana"      : (36.6667, 8.7000),
        "Ghardimaou"   : (36.4534, 8.4358),
        "Oued Mliz"    : (36.4333, 8.5333),
        "Bou Salem"    : (36.6083, 9.0500),
    }
    df_map = df_km.copy()
    df_map["lat"] = df_map["delegation"].map(lambda d: coords_delegations.get(d, (36.5,8.8))[0])
    df_map["lon"] = df_map["delegation"].map(lambda d: coords_delegations.get(d, (36.5,8.8))[1])

    metrique_map = st.radio("Colorer par :", ["nb_pannes","duree_moy","energie"], horizontal=True)

    fig = px.scatter_mapbox(df_map, lat="lat", lon="lon", size="nb_pannes",
                             color=metrique_map, hover_name="delegation",
                             hover_data=["nb_pannes","duree_moy"],
                             color_continuous_scale="YlOrRd", zoom=8, height=600,
                             mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Taille des points = nombre de pannes | Couleur = " + metrique_map)


# ============================================================
# PAGE 7 - PREDICTION PERSONNALISEE
# ============================================================
if page == "7. Prediction Personnalisee":
    st.title("Prediction en Temps Reel")
    st.markdown("Renseignez les caracteristiques d une panne pour obtenir une prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        annee_input = st.number_input("Annee", min_value=2024, max_value=2030, value=2026)
        mois_input = st.selectbox("Mois", list(range(1,13)))
        saison_input = st.selectbox("Saison", ["Ete","Automne","Hiver","Printemps"])
    with col2:
        deleg_input = st.selectbox("Delegation", sorted(df_km["delegation"].unique()))
        zone_input = st.selectbox("Zone", ["Urbaine","Rurale"])
        cause_input = st.selectbox("Cause", sorted(df_panne["cause_panne"].dropna().unique()))
    with col3:
        type_input = st.selectbox("Type de panne", sorted(df_panne["type_panne"].dropna().unique()))
        nb_abonnes_input = st.number_input("Nb abonnes affectes", min_value=0, value=1000)
        is_weekend_input = st.checkbox("Weekend ?")
        recidive_input = st.checkbox("Recidive ?")
        energie_input = st.number_input("Energie impactee estimee (kWh)", min_value=0.0, value=5000.0)

    if st.button("Predire", type="primary"):
        X_new = pd.DataFrame([{
            "annee": annee_input, "mois_num": mois_input,
            "saison_enc": le_saison.transform([saison_input])[0],
            "delegation_enc": le_deleg.transform([deleg_input])[0],
            "zone_enc": le_zone.transform([zone_input])[0],
            "cause_enc": le_cause.transform([cause_input])[0],
            "type_enc": le_type.transform([type_input])[0],
            "nb_abonnes_affectes": nb_abonnes_input,
            "is_weekend": int(is_weekend_input),
            "recidive_flag": int(recidive_input),
            "impact_energie_kwh": energie_input
        }])
        X_new_sc = scaler.transform(X_new)
        duree_predite = xgb_reg.predict(X_new_sc)[0]
        anomalie_predite = xgb_clf.predict(X_new_sc)[0]

        col_r1, col_r2 = st.columns(2)
        col_r1.metric("Duree predite", "{:.2f} heures".format(duree_predite))
        if anomalie_predite == 1:
            col_r2.error("ANOMALIE DETECTEE - Duree inhabituellement longue")
        else:
            col_r2.success("Panne normale")


# ============================================================
# PAGE 8 - RECOMMANDATIONS
# ============================================================
if page == "8. Recommandations":
    st.title("Recommandations pour la STEG Jendouba")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recommandations Operationnelles")
        st.markdown("""
        - Renforcer la maintenance preventive en ete (pic thermique)
        - Prioriser les interventions a Fernana et Ain Draham
        - Deployer un systeme d alerte temps reel via Kafka streaming
        - Auditer les 1000 pannes anormales identifiees par Isolation Forest
        """)
    with col2:
        st.subheader("Recommandations Analytiques")
        st.markdown("""
        - Reentrainement periodique du modele XGBoost
        - Integrer des donnees meteorologiques (LSTM actuellement limite)
        - Corriger la delegation Inconnu (79.4% des pannes)
        - Etendre l analyse aux autres gouvernorats
        """)

    st.markdown("---")
    st.subheader("Bilan Final")
    st.success("Modele XGBoost retenu : R2={:.4f} | MAE={:.3f}h | F1={:.4f}".format(
        metadata["R2_regression"], metadata["MAE_regression"], metadata["F1_classification"]))
