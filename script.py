import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


df = pd.read_csv('GlobalLandTemperaturesByCity.csv')
# Convertir les dates en datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Convertir les dates en datetime
df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

# Filtrer pour l'été (juin à août) à New York
targetedTime = df[
    (df["City"] == "New York") & (df["dt"].dt.year >= 2000) & (df["dt"].dt.month.isin([6, 7, 8]))
]

# Grouper par année et calculer la moyenne
df_summer_grouped = targetedTime.groupby(df["dt"].dt.year)["AverageTemperature"].mean().reset_index()

# Ajouter une colonne "Type" pour indiquer qu'il s'agit de vraies données
df_summer_grouped["Type"] = "Réel"

# Séparer les variables
X = df_summer_grouped["dt"].values.reshape(-1, 1)  # Années en colonne
y = df_summer_grouped["AverageTemperature"].values  # Températures

# Entraîner le modèle de régression
model = LinearRegression()
model.fit(X, y)

# Prédire jusqu'en 2020
future_years = np.arange(df_summer_grouped["dt"].max() + 1, 2021).reshape(-1, 1)
predictions = model.predict(future_years)

# Créer un DataFrame pour les prédictions
df_predictions = pd.DataFrame({"dt": future_years.flatten(), "AverageTemperature": predictions, "Type": "Prédiction"})

# Fusionner les vraies valeurs et les prédictions
df_combined = pd.concat([df_summer_grouped, df_predictions])

# --- 🎨 VISUALISATION --- #
plt.figure(figsize=(12, 6))

# Tracer les barres des valeurs réelles
sns.barplot(
    x=df_summer_grouped["dt"],
    y=df_summer_grouped["AverageTemperature"],
    color="blue",
    label="Données Réelles"
)

# Tracer les barres des prédictions
sns.barplot(
    x=df_predictions["dt"],
    y=df_predictions["AverageTemperature"],
    color="red",
    label="Prédictions"
)

# Amélioration du style
plt.xlabel("Années")
plt.ylabel("Température Moyenne (°C)")
plt.title("Prédiction des Températures Estivales à New York")
plt.legend()
plt.xticks(rotation=45)
plt.show()
