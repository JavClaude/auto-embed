# auto-embed

<div align="center">
<img src="logo/auto-embed-logo.png" width="256 "/>
</div>

Système de recommandations basé sur des embeddings classifiés utilisant un autoencodeur pour apprendre des représentations vectorielles de données catégorielles et numériques.

## 🎯 Vue d'ensemble

Ce projet implémente un système de recommandations qui :
- Entraîne un autoencodeur sur des données classifiées (catégorielles + numériques)
- Génère des embeddings vectoriels pour chaque élément classifié
- Stocke les embeddings dans ChromaDB pour une recherche de similarité efficace
- Fournit des recommandations basées sur la similarité vectorielle
- Suit les embeddings des utilisateurs basés sur leurs interactions

## 🚀 Installation

```bash
# Installation des dépendances core
make install-project
```

## 📋 Utilisation

### Entraînement du modèle
```bash
# Entraîner un modèle d'embeddings classifiés
train_recommendation_model --online_date 2025-06-23 --bottle_neck_size 32 --hidden_layer_sizes [128,64,32] --epochs 10 --batch_size 256
```

### Génération des embeddings
```bash
# Générer les embeddings pour un modèle entraîné
predict_recommendation_model --model_id <model_id> --date_to_predict 2025-06-23
```

### Obtenir des recommandations
```bash
# Obtenir des recommandations pour un élément classifié
what_is_my_recommendation --classified_ref <classified_ref>
```

### API REST
```bash
# Démarrer l'API de recommandations
start_recommendation_api
```

## 🏗️ Architecture

Le projet suit une architecture hexagonale avec :
- **Domain** : Entités métier et interfaces
- **Infrastructure** : Adaptateurs pour ChromaDB, Keras, stockage local
- **Use Cases** : Logique métier d'entraînement, prédiction et recommandations
- **API/CLI** : Points d'entrée pour l'interaction

## 🧠 Modèle

L'autoencodeur utilise :
- Couche d'embedding pour les variables catégorielles
- Couches denses pour l'encodage/décodage
- Couche goulot d'étranglement (bottleneck) pour la représentation compressée
- Fonction de perte multi-objectifs (MSE pour numérique, crossentropy pour catégoriel)

## 🛠️ Développement

```bash
# Linting
make run-lint

# Tests
make run-tests

# Tests avec couverture
make run-tests-coverage
```

## 📊 Stack technique

- **ML/DL** : TensorFlow/Keras, NumPy, Pandas
- **Vector DB** : ChromaDB
- **API** : FastAPI, Uvicorn
- **CLI** : Fire
- **DI** : Kink
- **Dev Tools** : Poetry, Ruff, Black, Pytest

