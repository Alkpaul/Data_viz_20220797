#!/usr/bin/env python3
"""
Enhanced Launch Script for French State Budget Analysis Application
Checks dependencies and launches the advanced Streamlit application
"""

import subprocess
import sys
import os

def check_file_exists(filename):
    """Vérifie si un fichier existe"""
    if not os.path.exists(filename):
        print(f"❌ Fichier manquant: {filename}")
        return False
    print(f"✅ Fichier trouvé: {filename}")
    return True

def install_requirements():
    """Installe les dépendances si nécessaire"""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        print("✅ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"⚠️  Dépendance manquante: {e}")
        print("Installation des dépendances...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dépendances installées avec succès")
            return True
        except subprocess.CalledProcessError:
            print("❌ Erreur lors de l'installation des dépendances")
            return False

def main():
    print("🚀 Lancement de l'Application d'Analyse Budgétaire")
    print("=" * 50)
    
    # Check required files
    required_files = [
        "app_historical.py",  # Use historical storytelling version
        "requirements.txt",
        "balances_clean.csv"
    ]
    
    all_files_exist = True
    for file in required_files:
        if not check_file_exists(file):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Certains fichiers sont manquants.")
        print("Assurez-vous d'avoir exécuté le notebook Jupyter pour générer les données.")
        return
    
    # Vérifier et installer les dépendances
    if not install_requirements():
        print("\n❌ Impossible d'installer les dépendances.")
        return
    
    print("\n🎯 Lancement de l'application Streamlit...")
    print("L'application sera accessible à l'adresse: http://localhost:8501")
    print("Appuyez sur Ctrl+C pour arrêter l'application")
    print("=" * 50)
    
    # Launch Streamlit with historical storytelling app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_historical.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée par l'utilisateur")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors du lancement: {e}")

if __name__ == "__main__":
    main()
