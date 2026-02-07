# 🎯 Système d'Analyse de Sentiments Client

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description

Système automatisé d'analyse de sentiments clients basé sur le Machine Learning et le NLP (Natural Language Processing). Ce projet permet de classifier automatiquement les avis clients en sentiments **Positifs** ou **Négatifs** avec une précision de **98,07%**.

### ✨ Fonctionnalités Principales

- 🤖 **Classification automatique** : Analyse de sentiments en temps réel
- 📊 **Dashboard interactif** : Visualisation des statistiques et tendances
- 📝 **Historique des prédictions** : Suivi complet de toutes les analyses
- 🔔 **Monitoring réactif** : Alertes automatiques si ratio négatif > seuil critique
- 🚀 **Performance optimale** : 98,07% d'accuracy avec Linear SVM
- ⚡ **Temps de réponse** : < 100ms par prédiction

### 🎓 Contexte Académique

Ce projet a été développé dans le cadre d'un projet de Machine Learning et NLP.

---

## 🏗️ Architecture du Projet

```
sentiment-analysis/
│
├── app.py                     # Application Flask principale
├── config.py                  # Configuration de l'app
├── run.py                     # Script de lancement
├── requirements.txt           # Dépendances Python
├── nixpacks.toml              # Configuration pour le déploiement (Railway/Render)
├── Procfile                   # Commande pour serveur Gunicorn
├── .gitattributes             # Configuration Git LFS (pour les .pkl)
│
├── Data/
│   └── raw/
│       └── processed/
│           └── vectorization_objects.pkl 
│
├── models/
│   └── final_optimized_model.pkl          
│
└── templates/                 # Dossier des vues HTML
    ├── base.html
    ├── index.html
    ├── history.html
    ├── stats.html
    └── about.html
```

---

## 🔧 Installation

### Prérequis

- **Python 3.8+** (recommandé : Python 3.9 ou 3.10)
- **pip** (gestionnaire de paquets Python)
- **virtualenv** (recommandé pour isolation)

### Étape 1 : Cloner le Repository

```bash
git clone https://github.com/votre-username/sentiment-analysis.git
cd sentiment-analysis
```

### Étape 2 : Créer un Environnement Virtuel

**Linux / macOS :**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

### Étape 3 : Installer les Dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **⚠️ Note importante** : Les packages `gensim` et `textblob` doivent être installés en dernier comme spécifié dans `requirements.txt`.

### Étape 4 : Télécharger les Ressources NLP

Après l'installation, téléchargez les ressources nécessaires pour TextBlob :

```bash
python -m textblob.download_corpora
```

Si vous utilisez NLTK directement, téléchargez également :

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### Étape 5 : Configuration (Optionnel)

Créez un fichier `.env` pour les variables d'environnement :

```bash
# .env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=votre_clé_secrète_ici
DATABASE_URI=sqlite:///data/predictions.db
```

---

## 🚀 Utilisation

### Démarrage en Mode Développement

```bash
python app.py
```

L'application sera accessible à l'adresse : **http://localhost:5000**

---

## 🔍 Pipeline de Traitement

### 1. Prétraitement NLP

```python
# Étapes automatisées :
1. Conversion en minuscules
2. Suppression ponctuation et caractères spéciaux
3. Tokenisation
4. Retrait des stopwords
5. Lemmatisation (WordNet)
6. Gestion des négations (not_good → single token)
```

### 2. Vectorisation TF-IDF

```python
# Paramètres optimisés :
- max_features: 10,000
- min_df: 5
- max_df: 0.7
- ngram_range: (1, 2)  # Unigrammes + Bigrammes
```

### 3. Classification Linear SVM

```python
# Hyperparamètres optimaux (GridSearchCV) :
- C: 1.0
- kernel: 'linear'
- class_weight: None
```

---

## 📦 Dépendances Principales

### Web Framework
- **Flask 3.1.2** : Framework web léger et flexible
- **Werkzeug 3.1.3** : WSGI utilities
- **Jinja2 3.1.6** : Moteur de templates
- **Gunicorn 21.2.0** : Serveur WSGI pour production

### Machine Learning
- **scikit-learn 1.6.1** : Bibliothèque ML principale
- **numpy 2.1.3** : Calculs numériques
- **pandas 2.2.3** : Manipulation de données
- **scipy 1.15.2** : Algorithmes scientifiques
- **joblib 1.4.2** : Sérialisation des modèles

### NLP (Natural Language Processing)
- **gensim** : Word2Vec et modèles de topics
- **textblob** : Analyse de sentiments et POS tagging

### Utilitaires
- **requests 2.32.3** : Requêtes HTTP
- **python-dateutil 2.9.0** : Manipulation de dates
- **pytz 2025.1** : Gestion des fuseaux horaires

---

## 👥 Auteurs & Contributeurs

Ce projet est le fruit d'une collaboration passionnée entre :

* **Akilou Illa Abdourazak** : [Abdourazak01](https://github.com/Abdourazak01)
* **Yassine Boumhand** : [Yboumhand](https://github.com/Yboumhand)

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. **Fork** le projet
2. Créez une **branche** pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. **Committez** vos changements (`git commit -m 'Add AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une **Pull Request**
---

## ⭐ Si ce projet vous a été utile

N'hésitez pas à lui donner une ⭐ sur GitHub !

---

**Dernière mise à jour** : 7 Février 2025
**Version** : 1.0.0
