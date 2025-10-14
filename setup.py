"""
Script de configuration pour le projet d'analyse budgétaire
Génère les fichiers nécessaires si ils n'existent pas
"""

import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Crée des données d'exemple si le fichier principal n'existe pas"""
    print("📊 Création de données d'exemple...")
    
    # Données d'exemple basées sur la structure réelle
    sample_data = {
        'Postes': ['Produits régaliens nets', 'Dépenses de personnel', 'Dépenses de fonctionnement', 'Dépenses d\'investissement'] * 100,
        'Sous-postes': ['Impôt sur le revenu', 'Salaires', 'Fournitures', 'Immobilisations'] * 100,
        'Indicateurs de synthèse': ['R001_Impôt sur le revenu', 'D001_Salaires', 'D002_Fournitures', 'D003_Immobilisations'] * 100,
        'Indicateurs de détail': ['Détail 1', 'Détail 2', 'Détail 3', 'Détail 4'] * 100,
        'Compte': [7711000000, 6411000000, 6011000000, 2011000000] * 100,
        'Nature Budgétaire': ['110102', '110104', '130105', '130106'] * 100,
        'Programme': ['0156', 'RBG', 'RBG', '0200'] * 100,
        'Libellé Ministère': ['Économie, finances et relance', 'Éducation nationale', 'Santé', 'Défense'] * 100,
        'Balance Sortie 2022': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2021': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2020': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2019': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2018': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2017': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2016': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2015': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2014': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2013': np.random.normal(0, 1000000, 400),
        'Balance Sortie 2012': np.random.normal(0, 1000000, 400)
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('balances_clean.csv', index=False, encoding='utf-8')
    print("✅ Fichier balances_clean.csv créé avec des données d'exemple")

def create_data_dictionary():
    """Crée un dictionnaire de données basique"""
    print("📚 Création du dictionnaire de données...")
    
    dict_data = {
        'colonne': ['Postes', 'Sous-postes', 'Indicateurs de synthèse', 'Balance Sortie 2022'],
        'type': ['object', 'object', 'object', 'float64'],
        'non_nuls': [400, 400, 400, 400],
        'nuls': [0, 0, 0, 0],
        'exemples': ['Produits régaliens nets', 'Impôt sur le revenu', 'R001_Impôt sur le revenu', '1234567.89']
    }
    
    df_dict = pd.DataFrame(dict_data)
    df_dict.to_csv('data_dictionary_clean.csv', index=False, encoding='utf-8')
    print("✅ Dictionnaire de données créé")

def main():
    print("🔧 Configuration du projet d'analyse budgétaire")
    print("=" * 50)
    
    # Vérifier si les fichiers existent
    if not os.path.exists('balances_clean.csv'):
        create_sample_data()
    else:
        print("✅ Fichier balances_clean.csv existe déjà")
    
    if not os.path.exists('data_dictionary_clean.csv'):
        create_data_dictionary()
    else:
        print("✅ Dictionnaire de données existe déjà")
    
    print("\n🎯 Configuration terminée !")
    print("Vous pouvez maintenant lancer l'application avec: python run_app.py")

if __name__ == "__main__":
    main()
