# =============================================================================
# FRENCH STATE BUDGET ANALYSIS WEB APPLICATION
# =============================================================================
# This application provides comprehensive analysis and visualization of French
# state budget data from 2012-2022. It includes advanced analytics, predictive
# modeling, and interactive dashboards for budget analysis.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Configure the Streamlit page with wide layout and expanded sidebar for
# better user experience and more space for visualizations
st.set_page_config(
    page_title="📊 French State Budget Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
# Advanced CSS styling for modern, professional appearance with custom
# colors, animations, and responsive design elements
st.markdown("""
<style>
    /* Main header styling with gradient background */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhanced metric cards with hover effects */
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
        border-right: 2px solid #dee2e6;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #2e86ab);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2e86ab, #1f77b4);
        transform: scale(1.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(90deg, #d4edda, #c3e6cb);
        border-left: 4px solid #28a745;
    }
    
    .stError {
        background: linear-gradient(90deg, #f8d7da, #f5c6cb);
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charge et prépare les données"""
    try:
        # Charger les données nettoyées
        df = pd.read_csv('balances_clean.csv', encoding='utf-8')
        
        # Fonction pour nettoyer les valeurs de balance
        def clean_balance_value(value):
            if pd.isna(value) or value == '-' or str(value).strip() == '':
                return np.nan
            cleaned = str(value).strip().replace(' ', '').replace(',', '.')
            try:
                return float(cleaned)
            except ValueError:
                return np.nan
        
        # Nettoyer les colonnes de balance
        balance_columns = [col for col in df.columns if 'Balance Sortie' in col]
        for col in balance_columns:
            df[col] = df[col].apply(clean_balance_value)
        
        # Créer le format long
        df_melted = pd.melt(
            df, 
            id_vars=['Postes', 'Sous-postes', 'Indicateurs de synthèse', 'Indicateurs de détail', 
                     'Compte', 'Nature Budgétaire', 'Programme', 'Libellé Ministère'],
            value_vars=balance_columns,
            var_name='Année',
            value_name='Balance'
        )
        
        # Extraire l'année
        df_melted['Année'] = df_melted['Année'].str.extract(r'(\d{4})').astype(int)
        
        return df, df_melted, balance_columns
    except FileNotFoundError:
        st.error("Fichier de données non trouvé. Assurez-vous que 'balances_clean.csv' est présent.")
        return None, None, None

def main():
    # Header principal
    st.markdown('<h1 class="main-header">📊 Analyse Budgétaire de l\'État Français</h1>', unsafe_allow_html=True)
    st.markdown("### Données des balances des comptes de l'État (2012-2022)")
    
    # Chargement des données
    df, df_melted, balance_columns = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar pour les filtres
    st.sidebar.header("🔧 Filtres et Options")
    
    # Filtre par année
    years = sorted(df_melted['Année'].unique())
    selected_years = st.sidebar.multiselect(
        "Sélectionner les années",
        options=years,
        default=years,
        help="Choisissez les années à analyser"
    )
    
    # Filtre par ministère
    ministeres = sorted([m for m in df_melted['Libellé Ministère'].unique() if pd.notna(m)])
    selected_ministeres = st.sidebar.multiselect(
        "Sélectionner les ministères",
        options=ministeres,
        default=ministeres[:5],  # Top 5 par défaut
        help="Choisissez les ministères à analyser"
    )
    
    # Filtre par poste
    postes = sorted(df_melted['Postes'].unique())
    selected_postes = st.sidebar.multiselect(
        "Sélectionner les postes budgétaires",
        options=postes,
        default=postes[:10],  # Top 10 par défaut
        help="Choisissez les postes budgétaires à analyser"
    )
    
    # Appliquer les filtres
    df_filtered = df_melted[
        (df_melted['Année'].isin(selected_years)) &
        (df_melted['Libellé Ministère'].isin(selected_ministeres)) &
        (df_melted['Postes'].isin(selected_postes))
    ]
    
    # Métriques principales
    st.header("📈 Métriques Principales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_balance = df_filtered['Balance'].sum() / 1e6
        st.metric(
            label="Balance Totale",
            value=f"{total_balance:,.0f} M€",
            delta=f"{total_balance/len(selected_years):,.0f} M€/an" if selected_years else "0 M€/an"
        )
    
    with col2:
        avg_balance = df_filtered['Balance'].mean()
        st.metric(
            label="Balance Moyenne",
            value=f"{avg_balance:,.0f} €",
            delta="Par entrée"
        )
    
    with col3:
        positive_entries = (df_filtered['Balance'] > 0).sum()
        total_entries = len(df_filtered)
        st.metric(
            label="Entrées Positives",
            value=f"{positive_entries:,}",
            delta=f"{(positive_entries/total_entries*100):.1f}%" if total_entries > 0 else "0%"
        )
    
    with col4:
        unique_postes = df_filtered['Postes'].nunique()
        st.metric(
            label="Postes Uniques",
            value=f"{unique_postes:,}",
            delta="Dans la sélection"
        )
    
    # Onglets pour les différentes analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'ensemble", 
        "🏛️ Analyse par Ministères", 
        "💰 Analyse par Postes", 
        "📈 Évolution Temporelle",
        "🔍 Analyse Détaillée"
    ])
    
    with tab1:
        st.header("Vue d'ensemble des données")
        
        # Graphique 1: Évolution temporelle globale
        balance_by_year = df_filtered.groupby('Année')['Balance'].sum() / 1e6
        
        fig1 = px.line(
            x=balance_by_year.index,
            y=balance_by_year.values,
            title="Évolution de la Balance Totale par Année",
            labels={'x': 'Année', 'y': 'Balance (Millions d\'euros)'},
            markers=True
        )
        fig1.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Équilibre")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Graphique 2: Distribution des balances
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = px.histogram(
                df_filtered,
                x='Balance',
                nbins=50,
                title="Distribution des Balances",
                labels={'Balance': 'Balance (Euros)', 'count': 'Nombre d\'entrées'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Top 10 des postes
            top_postes = df_filtered.groupby('Postes')['Balance'].sum().sort_values(ascending=False).head(10)
            fig3 = px.bar(
                x=top_postes.values / 1e6,
                y=top_postes.index,
                orientation='h',
                title="Top 10 des Postes par Balance Totale",
                labels={'x': 'Balance (Millions d\'euros)', 'y': 'Postes'}
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.header("Analyse par Ministères")
        
        # Graphique des ministères
        ministere_totals = df_filtered.groupby('Libellé Ministère')['Balance'].sum().sort_values(ascending=False)
        
        fig4 = px.bar(
            x=ministere_totals.values / 1e6,
            y=ministere_totals.index,
            orientation='h',
            title="Balance Totale par Ministère",
            labels={'x': 'Balance (Millions d\'euros)', 'y': 'Ministères'},
            color=ministere_totals.values,
            color_continuous_scale='RdYlBu_r'
        )
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Évolution des ministères dans le temps
        if len(selected_ministeres) <= 5:  # Limiter pour la lisibilité
            ministere_evolution = df_filtered.groupby(['Année', 'Libellé Ministère'])['Balance'].sum().unstack(fill_value=0)
            
            fig5 = go.Figure()
            for ministere in ministere_evolution.columns:
                fig5.add_trace(go.Scatter(
                    x=ministere_evolution.index,
                    y=ministere_evolution[ministere] / 1e6,
                    mode='lines+markers',
                    name=ministere,
                    line=dict(width=3)
                ))
            
            fig5.update_layout(
                title="Évolution des Ministères dans le Temps",
                xaxis_title="Année",
                yaxis_title="Balance (Millions d'euros)",
                height=500
            )
            st.plotly_chart(fig5, use_container_width=True)
    
    with tab3:
        st.header("Analyse par Postes Budgétaires")
        
        # Heatmap des postes par année
        if len(selected_postes) <= 15:  # Limiter pour la lisibilité
            poste_evolution = df_filtered.groupby(['Année', 'Postes'])['Balance'].sum().unstack(fill_value=0)
            
            fig6 = px.imshow(
                poste_evolution.T / 1e6,
                title="Heatmap des Postes par Année (Millions d'euros)",
                labels=dict(x="Année", y="Postes", color="Balance (M€)"),
                aspect="auto",
                color_continuous_scale='RdBu_r'
            )
            fig6.update_layout(height=600)
            st.plotly_chart(fig6, use_container_width=True)
        
        # Analyse des recettes vs dépenses
        col1, col2 = st.columns(2)
        
        with col1:
            recettes = df_filtered[df_filtered['Balance'] > 0].groupby('Postes')['Balance'].sum().sort_values(ascending=False).head(10)
            fig7 = px.bar(
                x=recettes.values / 1e6,
                y=recettes.index,
                orientation='h',
                title="Top 10 des Postes Recettes",
                labels={'x': 'Balance (Millions d\'euros)', 'y': 'Postes'},
                color=recettes.values,
                color_continuous_scale='Greens'
            )
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            depenses = df_filtered[df_filtered['Balance'] < 0].groupby('Postes')['Balance'].sum().sort_values(ascending=True).head(10)
            fig8 = px.bar(
                x=depenses.values / 1e6,
                y=depenses.index,
                orientation='h',
                title="Top 10 des Postes Dépenses",
                labels={'x': 'Balance (Millions d\'euros)', 'y': 'Postes'},
                color=depenses.values,
                color_continuous_scale='Reds'
            )
            fig8.update_layout(height=400)
            st.plotly_chart(fig8, use_container_width=True)
    
    with tab4:
        st.header("Évolution Temporelle")
        
        # Graphique interactif avec sélection
        st.subheader("Sélectionner les éléments à visualiser")
        
        col1, col2 = st.columns(2)
        with col1:
            show_ministeres = st.checkbox("Afficher les ministères", value=True)
        with col2:
            show_postes = st.checkbox("Afficher les postes", value=False)
        
        fig9 = go.Figure()
        
        if show_ministeres:
            for ministere in selected_ministeres[:5]:  # Limiter à 5 pour la lisibilité
                data_ministere = df_filtered[df_filtered['Libellé Ministère'] == ministere]
                evolution = data_ministere.groupby('Année')['Balance'].sum() / 1e6
                fig9.add_trace(go.Scatter(
                    x=evolution.index,
                    y=evolution.values,
                    mode='lines+markers',
                    name=f"Ministère: {ministere}",
                    line=dict(width=2)
                ))
        
        if show_postes:
            for poste in selected_postes[:5]:  # Limiter à 5 pour la lisibilité
                data_poste = df_filtered[df_filtered['Postes'] == poste]
                evolution = data_poste.groupby('Année')['Balance'].sum() / 1e6
                fig9.add_trace(go.Scatter(
                    x=evolution.index,
                    y=evolution.values,
                    mode='lines+markers',
                    name=f"Poste: {poste}",
                    line=dict(width=2, dash='dash')
                ))
        
        fig9.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Équilibre")
        fig9.update_layout(
            title="Évolution Temporelle Détaillée",
            xaxis_title="Année",
            yaxis_title="Balance (Millions d'euros)",
            height=500
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    with tab5:
        st.header("Analyse Détaillée")
        
        # Tableau de données interactif
        st.subheader("Données Détaillées")
        
        # Options d'affichage
        col1, col2, col3 = st.columns(3)
        with col1:
            show_columns = st.multiselect(
                "Colonnes à afficher",
                options=df_filtered.columns.tolist(),
                default=['Année', 'Postes', 'Libellé Ministère', 'Balance']
            )
        with col2:
            sort_by = st.selectbox("Trier par", options=df_filtered.columns.tolist(), index=len(df_filtered.columns)-1)
        with col3:
            sort_order = st.selectbox("Ordre", options=['Croissant', 'Décroissant'])
        
        # Appliquer les filtres
        df_display = df_filtered[show_columns].copy()
        ascending = sort_order == 'Croissant'
        df_display = df_display.sort_values(sort_by, ascending=ascending)
        
        # Afficher le tableau
        st.dataframe(
            df_display.head(1000),  # Limiter à 1000 lignes
            use_container_width=True,
            height=400
        )
        
        # Statistiques descriptives
        st.subheader("Statistiques Descriptives")
        if 'Balance' in df_display.columns:
            st.dataframe(df_display['Balance'].describe(), use_container_width=True)
        
        # Export des données
        st.subheader("Export des Données")
        csv = df_display.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="Télécharger les données filtrées (CSV)",
            data=csv,
            file_name=f"donnees_budgetaires_filtrees_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        📊 Application d'analyse budgétaire - Données des balances des comptes de l'État français (2012-2022)<br>
        Développée avec Streamlit et Plotly
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
