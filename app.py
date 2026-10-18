import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import json

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="STEG Jendouba - Prediction Pannes",
    layout="wide",
    page_icon="⚡"
)

# ============================================================
# STYLE CSS
# ============================================================

st.markdown("""
<style>

h1 {
    color: #1A237E;
    font-weight: 800;
    border-bottom: 3px solid #1E88E5;
    padding-bottom: 8px;
}

h2 {
    color: #1A237E;
    font-weight: 700;
    margin-top: 1.5rem;
}

h3 {
    color: #283593;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: #0D1B4C;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
}

[data-testid="stMetric"] {
    background-color: #F5F7FA;
    border: 1px solid #E0E0E0;
    border-left: 5px solid #1E88E5;
    border-radius: 10px;
    padding: 15px 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #455A64;
}

[data-testid="stMetricValue"] {
    color: #1A237E;
    font-weight: 800;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
}

.stButton > button {
    background-color: #1E88E5;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1565C0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

@st.cache_data
def load_data():

    df_panne = pd.read_csv(
        "data/pannes.csv",
        encoding="utf-8-sig"
    )

    df_fact = pd.read_csv(
        "data/facturation.csv",
        encoding="utf-8-sig"
    )

    df_km = pd.read_csv(
        "data/delegations_clusters.csv",
        encoding="utf-8-sig"
    )

 df_comp_reg = pd.read_csv(
    "data/comparatif_regression.csv",
    encoding="utf-8-sig"
)

# Supprimer définitivement LSTM du tableau comparatif
df_comp_reg = df_comp_reg[
    df_comp_reg["Algorithme"].str.strip().str.upper() != "LSTM"
].copy()

    df_comp_clf = pd.read_csv(
        "data/comparatif_classification.csv",
        encoding="utf-8-sig"
    )

    df_importance = pd.read_csv(
        "data/importance_variables.csv",
        index_col=0,
        encoding="utf-8-sig"
    )

    df_anomalies = pd.read_csv(
        "data/anomalies_pannes.csv",
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # SUPPRESSION DE LSTM DU TABLEAU COMPARATIF
    # --------------------------------------------------------

    if "Algorithme" in df_comp_reg.columns:
        df_comp_reg = df_comp_reg[
            df_comp_reg["Algorithme"].astype(str).str.upper() != "LSTM"
        ].copy()

    return (
        df_panne,
        df_fact,
        df_km,
        df_comp_reg,
        df_comp_clf,
        df_importance,
        df_anomalies
    )


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

@st.cache_resource
def load_model():

    # --------------------------------------------------------
    # MODELE DE REGRESSION
    # --------------------------------------------------------

    with open(
        "model/xgb_regression.pkl",
        "rb"
    ) as f:
        xgb_reg = pickle.load(f)

    # --------------------------------------------------------
    # MODELE DE CLASSIFICATION
    # --------------------------------------------------------

    with open(
        "model/xgb_classification.pkl",
        "rb"
    ) as f:
        xgb_clf = pickle.load(f)

    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------

    with open(
        "model/scaler.pkl",
        "rb"
    ) as f:
        scaler = pickle.load(f)

    # --------------------------------------------------------
    # ENCODEURS
    # --------------------------------------------------------

    with open(
        "model/label_encoder_saison.pkl",
        "rb"
    ) as f:
        le_saison = pickle.load(f)

    with open(
        "model/label_encoder_delegation.pkl",
        "rb"
    ) as f:
        le_deleg = pickle.load(f)

    with open(
        "model/label_encoder_zone.pkl",
        "rb"
    ) as f:
        le_zone = pickle.load(f)

    with open(
        "model/label_encoder_cause.pkl",
        "rb"
    ) as f:
        le_cause = pickle.load(f)

    with open(
        "model/label_encoder_type.pkl",
        "rb"
    ) as f:
        le_type = pickle.load(f)

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    with open(
        "model/metadata.json",
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    return (
        xgb_reg,
        xgb_clf,
        scaler,
        le_saison,
        le_deleg,
        le_zone,
        le_cause,
        le_type,
        metadata
    )


# ============================================================
# INITIALISATION
# ============================================================

(
    df_panne,
    df_fact,
    df_km,
    df_comp_reg,
    df_comp_clf,
    df_importance,
    df_anomalies
) = load_data()


(
    xgb_reg,
    xgb_clf,
    scaler,
    le_saison,
    le_deleg,
    le_zone,
    le_cause,
    le_type,
    metadata
) = load_model()


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("STEG Jendouba")

st.sidebar.markdown(
    "Prediction et Analyse des Pannes Electriques"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Tableau de Bord",
        "2. Donnees et Exploration",
        "3. Preparation des Donnees",
        "4. Modelisation et Evaluation",
        "5. Clustering et Anomalies",
        "6. Carte Interactive",
        "7. Prediction Personnalisee",
        "8. Recommandations"
    ]
)


# ============================================================
# PAGE 1 - TABLEAU DE BORD
# ============================================================

if page == "1. Tableau de Bord":

    st.title(
        "Tableau de Bord - STEG Jendouba"
    )

    st.markdown(
        "Vue synthetique du projet de prediction "
        "et analyse des pannes electriques"
    )

    # --------------------------------------------------------
    # INDICATEURS GENERAUX
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Pannes",
        "{:,}".format(len(df_panne))
    )

    nb_delegations = df_panne[
        df_panne["delegation"] != "Inconnu"
    ]["delegation"].nunique()

    col2.metric(
        "Delegations analysees",
        nb_delegations
    )

    col3.metric(
        "Duree moyenne",
        "{:.1f} h".format(
            df_panne["duree_h"].mean()
        )
    )

    col4.metric(
        "Anomalies detectees",
        "{:,}".format(
            int(df_anomalies["anomalie"].sum())
        )
    )

    # --------------------------------------------------------
    # PERFORMANCES
    # --------------------------------------------------------

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "R2 Regression (XGBoost)",
        "{:.3f}".format(
            metadata["R2_regression"]
        )
    )

    col6.metric(
        "F1 Classification (XGBoost)",
        "{:.3f}".format(
            metadata["F1_classification"]
        )
    )

    col7.metric(
        "MAE",
        "{:.3f} h".format(
            metadata["MAE_regression"]
        )
    )

    st.markdown("---")

    # --------------------------------------------------------
    # EVOLUTION ANNUELLE
    # --------------------------------------------------------

    col_a, col_b = st.columns(2)

    with col_a:

        st.subheader(
            "Evolution des pannes par annee"
        )

        evol = (
            df_panne
            .groupby("annee")
            .size()
            .reset_index(name="nb_pannes")
        )

        fig = px.line(
            evol,
            x="annee",
            y="nb_pannes",
            markers=True,
            title="Evolution annuelle du nombre de pannes"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # REPARTITION SAISONNIERE
    # --------------------------------------------------------

    with col_b:

        st.subheader(
            "Repartition par saison"
        )

        ordre = [
            "Ete",
            "Automne",
            "Hiver",
            "Printemps"
        ]

        saison_counts = (
            df_panne["saison"]
            .value_counts()
            .reindex(ordre)
            .fillna(0)
        )

        fig2 = px.pie(
            values=saison_counts.values,
            names=saison_counts.index,
            title="Repartition des pannes par saison"
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # VALIDATION CRISP-DM
    # --------------------------------------------------------

    st.subheader(
        "Validation des criteres metier CRISP-DM"
    )

    c1, c2, c3 = st.columns(3)

    if metadata["R2_regression"] > 0.70:
        c1.success(
            "R2 > 0.70 : VALIDE ({:.4f})".format(
                metadata["R2_regression"]
            )
        )
    else:
        c1.error(
            "R2 > 0.70 : NON VALIDE ({:.4f})".format(
                metadata["R2_regression"]
            )
        )

    if metadata["MAE_regression"] < 1:
        c2.success(
            "MAE < 1h : VALIDE ({:.3f} h)".format(
                metadata["MAE_regression"]
            )
        )
    else:
        c2.error(
            "MAE < 1h : NON VALIDE ({:.3f} h)".format(
                metadata["MAE_regression"]
            )
        )

    if metadata["F1_classification"] > 0.70:
        c3.success(
            "F1 > 0.70 : VALIDE ({:.4f})".format(
                metadata["F1_classification"]
            )
        )
    else:
        c3.error(
            "F1 > 0.70 : NON VALIDE ({:.4f})".format(
                metadata["F1_classification"]
            )
        )


# ============================================================
# PAGE 2 - DONNEES ET EXPLORATION
# ============================================================

if page == "2. Donnees et Exploration":

    st.title(
        "Donnees et Exploration"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Distribution",
            "Bivariee",
            "Regionale"
        ]
    )

    # --------------------------------------------------------
    # TAB 1 - DISTRIBUTION
    # --------------------------------------------------------

    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            ax.hist(
                df_panne["duree_h"].dropna(),
                bins=40,
                color="#E53935",
                edgecolor="white"
            )

            ax.axvline(
                df_panne["duree_h"].mean(),
                color="navy",
                linestyle="--",
                label="Moyenne"
            )

            ax.set_title(
                "Distribution Duree des Pannes"
            )

            ax.set_xlabel(
                "Duree (heures)"
            )

            ax.set_ylabel(
                "Frequence"
            )

            ax.legend()

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            fig2, ax2 = plt.subplots(
                figsize=(6, 4)
            )

            ax2.boxplot(
                df_panne["duree_h"].dropna(),
                vert=False
            )

            ax2.set_title(
                "Boxplot Duree des Pannes"
            )

            ax2.set_xlabel(
                "Duree (heures)"
            )

            st.pyplot(fig2)

            plt.close(fig2)

    # --------------------------------------------------------
    # TAB 2 - ANALYSE BIVARIEE
    # --------------------------------------------------------

    with tab2:

        col3, col4 = st.columns(2)

        with col3:

            df_sample = df_panne.sample(
                min(2000, len(df_panne)),
                random_state=41
            )

            fig3 = px.scatter(
                df_sample,
                x="duree_h",
                y="impact_energie_kwh",
                opacity=0.4,
                title="Duree vs Energie Perdue"
            )

            st.plotly_chart(
                fig3,
                width="stretch"
            )

        with col4:

            cols_corr = [
                "duree_h",
                "nb_abonnes_affectes",
                "impact_energie_kwh",
                "cout_estime_tnd"
            ]

            corr = df_panne[
                cols_corr
            ].corr()

            fig4, ax4 = plt.subplots(
                figsize=(6, 5)
            )

            sns.heatmap(
                corr,
                annot=True,
                cmap="RdYlBu_r",
                ax=ax4
            )

            ax4.set_title(
                "Matrice de Correlation"
            )

            st.pyplot(fig4)

            plt.close(fig4)

    # --------------------------------------------------------
    # TAB 3 - ANALYSE REGIONALE
    # --------------------------------------------------------

    with tab3:

        df_p = df_panne[
            df_panne["delegation"] != "Inconnu"
        ].copy()

        deleg_stats = (
            df_p
            .groupby("delegation")["duree_h"]
            .agg(["count", "mean"])
            .reset_index()
        )

        deleg_stats.columns = [
            "Delegation",
            "Nb Pannes",
            "Duree Moyenne"
        ]

        fig5 = px.bar(
            deleg_stats.sort_values(
                "Nb Pannes"
            ),
            x="Nb Pannes",
            y="Delegation",
            orientation="h",
            title="Pannes par Delegation"
        )

        st.plotly_chart(
            fig5,
            width="stretch"
        )

        st.dataframe(
            deleg_stats.sort_values(
                "Nb Pannes",
                ascending=False
            ),
            width="stretch"
        )


# ============================================================
# PAGE 3 - PREPARATION DES DONNEES
# ============================================================

if page == "3. Preparation des Donnees":

    st.title(
        "Preparation des Donnees - Feature Store"
    )

    tab1, tab2 = st.tabs(
        [
            "Feature Store",
            "Partitionnement"
        ]
    )

    # --------------------------------------------------------
    # TAB 1 - FEATURE STORE
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "11 Variables Explicatives"
        )

        features_info = pd.DataFrame({

            "Variable": [
                "annee",
                "mois_num",
                "saison_enc",
                "delegation_enc",
                "zone_enc",
                "cause_enc",
                "type_enc",
                "nb_abonnes_affectes",
                "is_weekend",
                "recidive_flag",
                "impact_energie_kwh"
            ],

            "Type": [
                "int",
                "int",
                "int",
                "int",
                "int",
                "int",
                "int",
                "float",
                "int",
                "int",
                "float"
            ],

            "Description": [
                "Annee incident",
                "Mois (1-12)",
                "Saison encodee",
                "Delegation encodee",
                "Zone encodee",
                "Cause encodee",
                "Type de panne encode",
                "Nombre d abonnes touches",
                "1 = weekend",
                "1 = recidive",
                "Energie non distribuee (kWh)"
            ]
        })

        st.dataframe(
            features_info,
            width="stretch"
        )

        st.info(
            "Variable cible regression : duree_h"
        )

        st.info(
            "Variable cible classification : anomalie_duree"
        )

    # --------------------------------------------------------
    # TAB 2 - PARTITIONNEMENT
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "Repartition Train / Test"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Lignes exploitables",
            "4 110"
        )

        col2.metric(
            "Train",
            "2 877 (70 %)"
        )

        col3.metric(
            "Test",
            "1 233 (30 %)"
        )

        st.metric(
            "random_state retenu",
            "41"
        )

        fig = px.pie(
            values=[70, 30],
            names=[
                "Train (70%)",
                "Test (30%)"
            ],
            title="Partitionnement des donnees"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# PAGE 4 - MODELISATION ET EVALUATION
# ============================================================

if page == "4. Modelisation et Evaluation":

    st.title(
        "Modelisation et Evaluation des Algorithmes"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Tableau Comparatif",
            "Visualisations",
            "XGBoost - Analyse"
        ]
    )

    # --------------------------------------------------------
    # TAB 1 - TABLEAU COMPARATIF
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "Regression - Prediction de duree_h"
        )

        st.dataframe(
            df_comp_reg.style
            .highlight_max(
                subset=["R2"],
                color="lightgreen"
            )
            .highlight_min(
                subset=["MAE", "RMSE"],
                color="lightgreen"
            ),
            width="stretch"
        )

        st.subheader(
            "Classification - Detection anomalie_duree"
        )

        st.dataframe(
            df_comp_clf.style
            .highlight_max(
                subset=["F1", "Accuracy"],
                color="lightgreen"
            ),
            width="stretch"
        )

    # --------------------------------------------------------
    # TAB 2 - VISUALISATIONS
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "Comparaison des performances de regression"
        )

        metrique = st.selectbox(
            "Choisir une metrique de regression",
            [
                "MAE",
                "RMSE",
                "R2"
            ]
        )

        fig = px.bar(
            df_comp_reg,
            x="Algorithme",
            y=metrique,
            color="Algorithme",
            title=f"Comparaison des algorithmes - {metrique}"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.subheader(
            "Comparaison des performances de classification"
        )

        metrique_clf = st.selectbox(
            "Choisir une metrique de classification",
            [
                "F1",
                "Accuracy",
                "Recall",
                "Precision"
            ]
        )

        fig_clf = px.bar(
            df_comp_clf,
            x="Algorithme",
            y=metrique_clf,
            color="Algorithme",
            title=f"Comparaison des algorithmes - {metrique_clf}"
        )

        st.plotly_chart(
            fig_clf,
            width="stretch"
        )

    # --------------------------------------------------------
    # TAB 3 - ANALYSE XGBOOST
    # --------------------------------------------------------

    with tab3:

        st.subheader(
            "Importance des Variables - XGBoost"
        )

        importance_sorted = (
            df_importance
            .sort_values(
                "XGBoost",
                ascending=True
            )
        )

        fig2 = px.bar(
            importance_sorted,
            x="XGBoost",
            y=importance_sorted.index,
            orientation="h",
            title="Importance des variables dans XGBoost"
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )

        st.markdown("---")

        st.subheader(
            "Modele de Regression"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "R2",
            "{:.3f}".format(
                metadata["R2_regression"]
            )
        )

        col2.metric(
            "MAE",
            "{:.3f} h".format(
                metadata["MAE_regression"]
            )
        )

        if "RMSE_regression" in metadata:
            col3.metric(
                "RMSE",
                "{:.3f} h".format(
                    metadata["RMSE_regression"]
                )
            )

        st.success(
            "XGBoost est utilise pour la prediction "
            "de la duree des pannes."
        )

        st.markdown("---")

        st.subheader(
            "Modele de Classification"
        )

        col4, col5 = st.columns(2)

        col4.metric(
            "F1-Score",
            "{:.3f}".format(
                metadata["F1_classification"]
            )
        )

        if "Accuracy_classification" in metadata:
            col5.metric(
                "Accuracy",
                "{:.3f}".format(
                    metadata["Accuracy_classification"]
                )
            )

        st.success(
            "XGBoost est utilise pour la detection "
            "des anomalies de duree."
        )


# ============================================================
# PAGE 5 - CLUSTERING ET ANOMALIES
# ============================================================

if page == "5. Clustering et Anomalies":

    st.title(
        "Clustering des Delegations et Detection des Anomalies"
    )

    tab1, tab2 = st.tabs(
        [
            "Clustering K-Means / DBSCAN",
            "Isolation Forest"
        ]
    )

    # --------------------------------------------------------
    # TAB 1 - CLUSTERING
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "Profils des Delegations"
        )

        colonnes_km = [
            "delegation",
            "cluster",
            "dbscan",
            "nb_pannes",
            "duree_moy"
        ]

        colonnes_km_existantes = [
            c for c in colonnes_km
            if c in df_km.columns
        ]

        st.dataframe(
            df_km[colonnes_km_existantes],
            width="stretch"
        )

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        if (
            "PC1" in df_km.columns
            and "PC2" in df_km.columns
        ):

            fig = px.scatter(
                df_km,
                x="PC1",
                y="PC2",
                color="cluster",
                text="delegation",
                title="Clusters K-Means - Vue PCA"
            )

            fig.update_traces(
                textposition="top center"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        st.info(
            "K-Means permet d'identifier des groupes de "
            "delegations ayant des profils similaires. "
            "DBSCAN complete cette analyse en identifiant "
            "les profils geographiques atypiques."
        )

    # --------------------------------------------------------
    # TAB 2 - ISOLATION FOREST
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "Detection des Pannes Anormales"
        )

        n_anom = int(
            df_anomalies["anomalie"].sum()
        )

        taux_anom = (
            n_anom /
            len(df_anomalies) *
            100
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Pannes analysees",
            f"{len(df_anomalies):,}"
        )

        col2.metric(
            "Pannes anormales detectees",
            f"{n_anom:,} ({taux_anom:.1f} %)"
        )

        st.subheader(
            "Causes des Pannes Anormales"
        )

        if "cause_panne" in df_anomalies.columns:

            top_causes = (
                df_anomalies[
                    df_anomalies["anomalie"] == 1
                ]["cause_panne"]
                .value_counts()
                .head(8)
            )

            fig2 = px.bar(
                x=top_causes.values,
                y=top_causes.index,
                orientation="h",
                title="Principales Causes des Pannes Anormales"
            )

            st.plotly_chart(
                fig2,
                width="stretch"
            )

        st.success(
            f"Isolation Forest a detecte "
            f"{n_anom:,} pannes anormales sur "
            f"{len(df_anomalies):,}, soit "
            f"{taux_anom:.1f} %."
        )


# ============================================================
# PAGE 6 - CARTE INTERACTIVE
# ============================================================

if page == "6. Carte Interactive":

    st.title(
        "Carte Interactive du Reseau Electrique - Jendouba"
    )

    # --------------------------------------------------------
    # COORDONNEES DES DELEGATIONS
    # --------------------------------------------------------

    coords_delegations = {

        "Jendouba": (
            36.5011,
            8.7802
        ),

        "Tabarka": (
            36.9547,
            8.7583
        ),

        "Ain Draham": (
            36.7794,
            8.6883
        ),

        "Fernana": (
            36.6667,
            8.7000
        ),

        "Ghardimaou": (
            36.4534,
            8.4358
        ),

        "Oued Mliz": (
            36.4333,
            8.5333
        ),

        "Bou Salem": (
            36.6083,
            9.0500
        )
    }

    df_map = df_km.copy()

    # --------------------------------------------------------
    # COORDONNEES
    # --------------------------------------------------------

    df_map["lat"] = df_map[
        "delegation"
    ].map(
        lambda d: coords_delegations.get(
            d,
            (36.5, 8.8)
        )[0]
    )

    df_map["lon"] = df_map[
        "delegation"
    ].map(
        lambda d: coords_delegations.get(
            d,
            (36.5, 8.8)
        )[1]
    )

    # --------------------------------------------------------
    # METRIQUES DISPONIBLES
    # --------------------------------------------------------

    metriques_disponibles = []

    if "nb_pannes" in df_map.columns:
        metriques_disponibles.append(
            "nb_pannes"
        )

    if "duree_moy" in df_map.columns:
        metriques_disponibles.append(
            "duree_moy"
        )

    if "energie" in df_map.columns:
        metriques_disponibles.append(
            "energie"
        )

    if not metriques_disponibles:

        st.error(
            "Aucune metrique geographique disponible "
            "dans delegations_clusters.csv."
        )

    else:

        metrique_map = st.radio(
            "Colorer les delegations selon :",
            metriques_disponibles,
            horizontal=True
        )

        # ----------------------------------------------------
        # CARTE
        # ----------------------------------------------------

        hover_columns = [
            c for c in [
                "nb_pannes",
                "duree_moy"
            ]
            if c in df_map.columns
        ]

        fig = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            size="nb_pannes"
            if "nb_pannes" in df_map.columns
            else None,
            color=metrique_map,
            hover_name="delegation",
            hover_data=hover_columns,
            color_continuous_scale="YlOrRd",
            zoom=8,
            height=600,
            mapbox_style="open-street-map"
        )

        fig.update_layout(
            margin=dict(
                l=0,
                r=0,
                t=40,
                b=0
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.info(
            "Taille des points = nombre de pannes | "
            "Couleur = " + metrique_map
        )


# ============================================================
# PAGE 7 - PREDICTION PERSONNALISEE
# ============================================================

if page == "7. Prediction Personnalisee":

    st.title(
        "Prediction en Temps Reel"
    )

    st.markdown(
        "Renseignez les caracteristiques d'une panne "
        "pour obtenir une prediction de sa duree "
        "et de son niveau d'anomalie."
    )

    # --------------------------------------------------------
    # SAISIE
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLONNE 1
    # --------------------------------------------------------

    with col1:

        annee_input = st.number_input(
            "Annee",
            min_value=2006,
            max_value=2030,
            value=2026
        )

        mois_input = st.selectbox(
            "Mois",
            list(range(1, 13))
        )

        saison_input = st.selectbox(
            "Saison",
            [
                "Ete",
                "Automne",
                "Hiver",
                "Printemps"
            ]
        )

    # --------------------------------------------------------
    # COLONNE 2
    # --------------------------------------------------------

    with col2:

        delegations_disponibles = sorted(
            df_km[
                "delegation"
            ]
            .dropna()
            .unique()
        )

        deleg_input = st.selectbox(
            "Delegation",
            delegations_disponibles
        )

        zones_disponibles = [
            "Urbaine",
            "Rurale"
        ]

        zone_input = st.selectbox(
            "Zone",
            zones_disponibles
        )

        causes_disponibles = sorted(
            df_panne[
                "cause_panne"
            ]
            .dropna()
            .unique()
        )

        cause_input = st.selectbox(
            "Cause",
            causes_disponibles
        )

    # --------------------------------------------------------
    # COLONNE 3
    # --------------------------------------------------------

    with col3:

        types_disponibles = sorted(
            df_panne[
                "type_panne"
            ]
            .dropna()
            .unique()
        )

        type_input = st.selectbox(
            "Type de panne",
            types_disponibles
        )

        nb_abonnes_input = st.number_input(
            "Nb abonnes affectes",
            min_value=0,
            value=1000
        )

        is_weekend_input = st.checkbox(
            "Weekend ?"
        )

        recidive_input = st.checkbox(
            "Recidive ?"
        )

        energie_input = st.number_input(
            "Energie impactee estimee (kWh)",
            min_value=0.0,
            value=5000.0
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "Predire",
        type="primary"
    ):

        try:

            # ------------------------------------------------
            # ENCODAGE DES VARIABLES
            # ------------------------------------------------

            saison_encoded = le_saison.transform(
                [saison_input]
            )[0]

            delegation_encoded = le_deleg.transform(
                [deleg_input]
            )[0]

            zone_encoded = le_zone.transform(
                [zone_input]
            )[0]

            cause_encoded = le_cause.transform(
                [cause_input]
            )[0]

            type_encoded = le_type.transform(
                [type_input]
            )[0]

            # ------------------------------------------------
            # CREATION DU VECTEUR DE PREDICTION
            # ------------------------------------------------

            X_new = pd.DataFrame([{

                "annee": annee_input,

                "mois_num": mois_input,

                "saison_enc": saison_encoded,

                "delegation_enc": delegation_encoded,

                "zone_enc": zone_encoded,

                "cause_enc": cause_encoded,

                "type_enc": type_encoded,

                "nb_abonnes_affectes":
                    nb_abonnes_input,

                "is_weekend":
                    int(is_weekend_input),

                "recidive_flag":
                    int(recidive_input),

                "impact_energie_kwh":
                    energie_input
            }])

            # ------------------------------------------------
            # NORMALISATION
            # ------------------------------------------------

            X_new_sc = scaler.transform(
                X_new
            )

            # ------------------------------------------------
            # PREDICTION DUREE
            # XGBOOST REGRESSION
            # ------------------------------------------------

            duree_predite = xgb_reg.predict(
                X_new_sc
            )[0]

            duree_predite = max(
                0,
                float(duree_predite)
            )

            # ------------------------------------------------
            # DETECTION ANOMALIE
            # XGBOOST CLASSIFICATION
            # ------------------------------------------------

            anomalie_predite = xgb_clf.predict(
                X_new_sc
            )[0]

            # ------------------------------------------------
            # PROBABILITE D'ANOMALIE
            # ------------------------------------------------

            probabilite_anomalie = None

            if hasattr(
                xgb_clf,
                "predict_proba"
            ):

                probabilite_anomalie = (
                    xgb_clf
                    .predict_proba(X_new_sc)[0][1]
                )

            # ------------------------------------------------
            # AFFICHAGE DES RESULTATS
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "Resultat de la prediction"
            )

            col_r1, col_r2, col_r3 = st.columns(3)

            col_r1.metric(
                "Duree predite",
                "{:.2f} heures".format(
                    duree_predite
                )
            )

            if anomalie_predite == 1:

                col_r2.error(
                    "ANOMALIE DETECTEE"
                )

                st.warning(
                    "La panne presente un profil "
                    "considere comme inhabituel."
                )

            else:

                col_r2.success(
                    "PANNE NORMALE"
                )

            if probabilite_anomalie is not None:

                col_r3.metric(
                    "Probabilite d'anomalie",
                    "{:.1f} %".format(
                        probabilite_anomalie * 100
                    )
                )

            # ------------------------------------------------
            # RESUME DE LA PANNE
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "Caracteristiques de la panne"
            )

            resume = pd.DataFrame({
                "Caracteristique": [
                    "Annee",
                    "Mois",
                    "Saison",
                    "Delegation",
                    "Zone",
                    "Cause",
                    "Type de panne",
                    "Abonnes affectes",
                    "Weekend",
                    "Recidive",
                    "Energie impactee"
                ],

                "Valeur": [
                    annee_input,
                    mois_input,
                    saison_input,
                    deleg_input,
                    zone_input,
                    cause_input,
                    type_input,
                    nb_abonnes_input,
                    "Oui"
                    if is_weekend_input
                    else "Non",
                    "Oui"
                    if recidive_input
                    else "Non",
                    f"{energie_input:.2f} kWh"
                ]
            })

            st.dataframe(
                resume,
                width="stretch",
                hide_index=True
            )

        except ValueError as e:

            st.error(
                "Erreur d'encodage d'une variable : "
                + str(e)
            )

        except Exception as e:

            st.error(
                "Erreur lors de la prediction : "
                + str(e)
            )


# ============================================================
# PAGE 8 - RECOMMANDATIONS
# ============================================================

if page == "8. Recommandations":

    st.title(
        "Recommandations pour la STEG Jendouba"
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # RECOMMANDATIONS OPERATIONNELLES
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Recommandations Operationnelles"
        )

        st.markdown("""
        - Renforcer la maintenance preventive en ete
          pendant les periodes de forte sollicitation.

        - Prioriser les interventions dans les delegations
          presentant les profils de pannes les plus critiques.

        - Exploiter les predictions de duree pour mieux
          planifier les equipes d'intervention.

        - Auditer les pannes anormales identifiees par
          le systeme de detection.

        - Utiliser les resultats du clustering pour
          orienter les actions de maintenance.
        """)

    # --------------------------------------------------------
    # RECOMMANDATIONS ANALYTIQUES
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "Recommandations Analytiques"
        )

        st.markdown("""
        - Reentrainer periodiquement les modeles
          XGBoost avec les nouvelles donnees disponibles.

        - Surveiller regulierement les performances
          de la regression et de la classification.

        - Integrer des donnees meteorologiques afin
          d'enrichir les variables explicatives.

        - Corriger progressivement les donnees associees
          a la delegation « Inconnu ».

        - Etendre ulterieurement l'analyse a d'autres
          gouvernorats.
        """)

    st.markdown("---")

    # --------------------------------------------------------
    # BILAN FINAL
    # --------------------------------------------------------

    st.subheader(
        "Bilan Final"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "XGBoost - R2",
        "{:.3f}".format(
            metadata["R2_regression"]
        )
    )

    col2.metric(
        "XGBoost - MAE",
        "{:.3f} h".format(
            metadata["MAE_regression"]
        )
    )

    col3.metric(
        "XGBoost - F1-Score",
        "{:.3f}".format(
            metadata["F1_classification"]
        )
    )

    st.success(
        "XGBoost est retenu pour la prediction de la "
        "duree des pannes ainsi que pour la detection "
        "des anomalies de duree."
    )
