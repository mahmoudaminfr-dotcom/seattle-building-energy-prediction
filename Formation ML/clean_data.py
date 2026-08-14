import pandas as pd
df = pd.read_csv('age_vs_poids_vs_taille_vs_sexe.csv'   )

# les variables prédictives
X = df[['sexe', 'age', 'taille']]

# la variable cible, le poids
y = df.poids

# on choisit un modèle de régression linéaire
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# on entraîne ce modèle sur les données avec la méthode fit
reg.fit(X, y)

# et on obtient directement un score
print(reg.score(X, y))

# ainsi que les coefficients a,b,c de la régression linéaire
print(reg.coef_)