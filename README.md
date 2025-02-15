# Projet de Régression Linéaire - Prédiction de Prix de Voitures

Ce projet implémente un modèle de régression linéaire pour prédire le prix des voitures en fonction de leur kilométrage. Il est composé de deux parties principales : un module d'entraînement du modèle et un module de prédiction.

## 📊 Vue d'ensemble du projet

Le projet utilise la régression linéaire, une technique d'apprentissage automatique fondamentale, pour établir une relation linéaire entre :
- Variable indépendante (X) : Kilométrage de la voiture
- Variable dépendante (y) : Prix de la voiture

## 🔧 Structure du Projet

Le projet est divisé en deux scripts principaux :

### 1. Module d'Entraînement (`training.py`)

Ce module implémente l'algorithme de descente de gradient pour entraîner le modèle :

```python
def train_model():
    X, y = load_data()                    # Chargement des données
    X_norm, X_mean, X_std = normalize_data(X)  # Normalisation
    theta_norm, cost_history = gradient_descent(X_norm, y_norm, 500)  # Entraînement
```

#### Fonctionnalités clés :
- Normalisation des données (z-score)
- Descente de gradient optimisée avec des opérations matricielles
- Calcul et suivi des métriques de performance (MSE, RMSE, R²)
- Sauvegarde automatique des paramètres du modèle

### 2. Module de Prédiction (`predict.py`)

Ce module permet d'utiliser le modèle entraîné pour faire des prédictions :

```python
def ft_predict():
    theta0, theta1 = ft_load_params('model_params.txt')
    # Prix = theta0 + theta1 * kilométrage
```

#### Fonctionnalités clés :
- Interface interactive pour les prédictions
- Validation robuste des entrées utilisateur
- Visualisation des prédictions
- Gestion des erreurs complète

## 🎯 Caractéristiques Techniques Importantes

### Validation des Données
Le projet inclut une validation exhaustive des données :
- Vérification des colonnes requises
- Détection des valeurs nulles
- Validation des valeurs négatives
- Vérification des formats de fichiers

### Normalisation
La normalisation z-score est utilisée pour standardiser les données :
```python
X_norm = (X - mean) / std
```
Cette étape est cruciale pour :
- Accélérer la convergence de la descente de gradient
- Améliorer la stabilité numérique
- Rendre le modèle plus robuste

### Descente de Gradient
L'implémentation utilise la descente de gradient par lots avec :
- Opérations matricielles optimisées avec NumPy
- Critère de convergence automatique
- Suivi de l'historique des coûts

### Visualisations
Le projet inclut plusieurs visualisations :
- Données d'entraînement et ligne de régression
- Historique de la fonction coût
- Points de prédiction individuels

## 📈 Métriques de Performance

Le modèle calcule plusieurs métriques pour évaluer sa performance :
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Précision du modèle basée sur l'écart-type

## 🚀 Utilisation

### Entraînement du modèle :
```bash
python training.py
```

### Faire des prédictions :
```bash
python predict.py
```

## 🛠 Dépendances
- NumPy : Calculs matriciels
- Pandas : Manipulation des données
- Matplotlib : Visualisations

## 🔍 Points Techniques Notables

1. **Robustesse** :
   - Gestion extensive des erreurs
   - Validation des données en entrée
   - Limites raisonnables sur les prédictions

2. **Performance** :
   - Utilisation d'opérations vectorisées
   - Critère de convergence optimisé
   - Normalisation des données

3. **Maintenabilité** :
   - Code modulaire
   - Documentation claire
   - Tests de validation intégrés

⚠️ Note

Vous l'avez remarqué, ce README a été généré par Claude AI. Le contenu a cependant été totalement vérifié par moi-même afin de correspondre parfaitement à mon implémentation. Enjoy 😊
