# 🚀 Guide de Déploiement

## Déploiement Local

### Prérequis
- Python 3.8+
- pip

### Installation
```bash
# 1. Cloner le repository
git clone <votre-repo>
cd Data_viz_20220797

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer les données (si nécessaire)
python setup.py

# 4. Lancer l'application
python run_app.py
# ou directement
streamlit run app.py
```

## Déploiement sur Streamlit Cloud

### Étapes
1. **Pousser le code sur GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connecter à Streamlit Cloud**
   - Aller sur [share.streamlit.io](https://share.streamlit.io)
   - Se connecter avec GitHub
   - Sélectionner le repository
   - Configurer l'application

3. **Configuration Streamlit Cloud**
   - **Main file path**: `app.py`
   - **Python version**: 3.8+
   - **Requirements file**: `requirements.txt`

### Fichiers requis pour Streamlit Cloud
- `app.py` (application principale)
- `requirements.txt` (dépendances)
- `balances_clean.csv` (données)
- `data_dictionary_clean.csv` (dictionnaire)

## Déploiement sur Heroku

### Configuration
1. **Créer `Procfile`**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Créer `runtime.txt`**
   ```
   python-3.9.16
   ```

3. **Déployer**
   ```bash
   # Installer Heroku CLI
   heroku create votre-app-name
   git push heroku main
   ```

## Déploiement avec Docker

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Commandes Docker
```bash
# Construire l'image
docker build -t budget-analysis-app .

# Lancer le container
docker run -p 8501:8501 budget-analysis-app
```

## Déploiement sur VPS/Serveur

### Configuration Nginx
```nginx
server {
    listen 80;
    server_name votre-domaine.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

### Service Systemd
```ini
[Unit]
Description=Budget Analysis App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/app
ExecStart=/usr/bin/streamlit run app.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

## Variables d'Environnement

### Configuration
```bash
# .env file
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Monitoring et Logs

### Logs Streamlit
```bash
# Voir les logs
streamlit run app.py --logger.level=debug

# Logs dans un fichier
streamlit run app.py > app.log 2>&1
```

### Monitoring avec PM2
```bash
# Installer PM2
npm install -g pm2

# Lancer l'application
pm2 start "streamlit run app.py" --name budget-app

# Monitoring
pm2 status
pm2 logs budget-app
```

## Sécurité

### Recommandations
- Utiliser HTTPS en production
- Configurer les CORS si nécessaire
- Limiter l'accès par IP si requis
- Utiliser des variables d'environnement pour les secrets

### Configuration HTTPS
```python
# Dans app.py
import ssl

# Configuration SSL
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('cert.pem', 'key.pem')
```

## Performance

### Optimisations
- Utiliser `@st.cache_data` pour les données
- Limiter le nombre de graphiques simultanés
- Optimiser les requêtes de données
- Utiliser des index sur les colonnes de filtrage

### Monitoring des Performances
```python
import time
import streamlit as st

@st.cache_data
def load_data():
    start_time = time.time()
    # Chargement des données
    end_time = time.time()
    st.write(f"Temps de chargement: {end_time - start_time:.2f}s")
    return data
```

## Troubleshooting

### Problèmes Courants
1. **Port déjà utilisé**
   ```bash
   streamlit run app.py --server.port=8502
   ```

2. **Erreur de dépendances**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Fichier de données manquant**
   ```bash
   python setup.py
   ```

### Logs d'Erreur
- Vérifier les logs Streamlit
- Contrôler les permissions de fichiers
- Vérifier la configuration réseau
