# house-price-prediction
# 🏠 Prédiction du Prix des Maisons avec Scikit-Learn

## 📌 Introduction
Ce projet consiste à construire un modèle de Machine Learning capable de prédire le prix des maisons en fonction de plusieurs caractéristiques (surface, nombre de chambres, localisation, etc.).

📊 **Données :** Dataset de prix de maisons
🤖 **Modèle :** Régression Linéaire avec Scikit-Learn
🚀 **Objectif :** Prédire les prix et optimiser le modèle pour réduire l'erreur

---

## 📂 Structure du Projet
```
house-price-prediction/
│── data/
│   ├── house_prices.csv  # Dataset
│── notebooks/
│   ├── data_analysis.ipynb  # Analyse et visualisation
│   ├── model_training.ipynb  # Entraînement du modèle
│── src/
│   ├── train_model.py  # Script pour entraîner le modèle
│── README.md  # Explication du projet
│── requirements.txt  # Bibliothèques requises
│── model.pkl  # Modèle sauvegardé
```

---

## 📊 1️⃣ Analyse Exploratoire des Données

📌 Charger et visualiser les données :
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/house_prices.csv")
print(df.head())

sns.pairplot(df)
plt.show()
```

---

## 🤖 2️⃣ Entraînement du Modèle

📌 Régression Linéaire avec Scikit-Learn :
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
```

---

## 💾 3️⃣ Sauvegarde du Modèle

📌 Sauvegarde du modèle avec `joblib` :
```python
import joblib
joblib.dump(model, "model.pkl")
```

---

## 🚀 4️⃣ Exécution du Projet

### **🔹 Installation des dépendances**
```bash
pip install -r requirements.txt
```

### **🔹 Exécution du script d'entraînement**
```bash
python src/train_model.py
```

---

## 📌 5️⃣ Améliorations Possibles
✅ Utiliser **Random Forest** ou **XGBoost** pour de meilleures prédictions.  
✅ Créer une **interface web** avec **Streamlit** pour interagir avec le modèle.  
✅ Ajouter une **optimisation des hyperparamètres** avec GridSearchCV.  

---

## 📌 6️⃣ Contribuer au Projet

Si vous souhaitez contribuer :
1️⃣ Forkez le repo 🔄
2️⃣ Créez une branche 💡
3️⃣ Ajoutez vos modifications 🛠
4️⃣ Faites un **Pull Request** ✅

---

## 📢 Contact
💬 N'hésitez pas à me contacter sur **LinkedIn** ou via GitHub !

---

🔥 **Si ce projet vous aide, n'oubliez pas de mettre une ★ sur GitHub !**

