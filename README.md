# 📊 Application d'Analyse Budgétaire de l'État Français

## 🎯 Description
Application web interactive pour analyser les données des balances des comptes de l'État français (2012-2022). Cette application permet d'explorer les données budgétaires de manière dynamique avec des visualisations interactives.

## 🚀 Installation et Lancement

### Prérequis
- Python 3.8+
- Les fichiers de données générés par le notebook Jupyter

### Installation
```bash
# Cloner le repository
git clone <votre-repo>
cd Data_viz_20220797

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement de l'application
```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

## 📋 Fonctionnalités

### 🔧 Filtres Interactifs
- **Sélection d'années** : Choisir les années à analyser (2012-2022)
- **Filtrage par ministères** : Sélectionner les ministères d'intérêt
- **Filtrage par postes budgétaires** : Choisir les postes à analyser

### 📊 Visualisations
1. **Vue d'ensemble**
   - Évolution temporelle de la balance totale
   - Distribution des balances
   - Top 10 des postes budgétaires

2. **Analyse par Ministères**
   - Balance totale par ministère
   - Évolution temporelle des ministères

3. **Analyse par Postes**
   - Heatmap des postes par année
   - Top des postes recettes vs dépenses

4. **Évolution Temporelle**
   - Graphiques interactifs avec sélection d'éléments
   - Comparaison ministères/postes

5. **Analyse Détaillée**
   - Tableau de données interactif
   - Statistiques descriptives
   - Export des données filtrées

### 📈 Métriques Principales
- Balance totale et moyenne
- Nombre d'entrées positives
- Nombre de postes uniques

## 🗂️ Structure des Fichiers

```
Data_viz_20220797/
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation
├── projet_dataviz_Briand_Paul_20220797.ipynb  # Notebook d'analyse
├── balances_clean.csv              # Données nettoyées (généré par le notebook)
├── data_dictionary_clean.csv       # Dictionnaire de données
├── data_dictionary_long.csv        # Dictionnaire format long
└── kpi_evolution_globale.png       # Figure exportée
```

## 🔍 Données

L'application utilise les données des balances des comptes de l'État français couvrant la période 2012-2022. Les données sont préprocessées dans le notebook Jupyter pour :

- Nettoyer les valeurs manquantes et les formats
- Convertir les types de données
- Restructurer au format long pour l'analyse temporelle

## 🎨 Interface

L'application propose une interface moderne et intuitive avec :
- **Sidebar** : Filtres et options
- **Onglets** : Organisation des différentes analyses
- **Graphiques interactifs** : Zoom, sélection, hover
- **Métriques en temps réel** : Mise à jour selon les filtres

## 🛠️ Technologies Utilisées

- **Streamlit** : Framework web pour l'interface
- **Plotly** : Visualisations interactives
- **Pandas** : Manipulation des données
- **NumPy** : Calculs numériques
- **Matplotlib/Seaborn** : Visualisations statiques

## 📝 Utilisation

1. **Lancer l'application** : `streamlit run app.py`
2. **Sélectionner les filtres** dans la sidebar
3. **Explorer les onglets** pour différentes analyses
4. **Interagir avec les graphiques** (zoom, sélection)
5. **Exporter les données** depuis l'onglet "Analyse Détaillée"

## 🔧 Personnalisation

L'application peut être facilement personnalisée :
- Modifier les couleurs dans le CSS
- Ajouter de nouveaux graphiques
- Intégrer d'autres sources de données
- Personnaliser les métriques

## 📊 Exemples d'Analyses

- **Tendance budgétaire** : Évolution globale des balances
- **Performance ministérielle** : Comparaison entre ministères
- **Structure budgétaire** : Répartition recettes/dépenses
- **Volatilité temporelle** : Identification des variations importantes

## 🚀 Déploiement

L'application peut être déployée sur :
- **Streamlit Cloud** : Déploiement gratuit
- **Heroku** : Avec les fichiers requirements.txt
- **Docker** : Containerisation pour la production

## 📞 Support

Pour toute question ou problème, consultez :
- La documentation Streamlit
- Le notebook Jupyter pour les détails du preprocessing
- Les commentaires dans le code source
