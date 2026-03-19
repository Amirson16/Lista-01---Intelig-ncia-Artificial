import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- ETAPA 1: CARREGAMENTO DO DATASET ---
df = pd.read_csv('marketing.csv')
print("Primeiras linhas do dataset:")
print(df.head())
print("\nInformações gerais:")
df.info()

# --- ETAPA 2: ANÁLISE EXPLORATÓRIA ---
plt.figure(figsize=(8, 5))
plt.scatter(df['anuncios'], df['cliques'], color='blue', alpha=0.5)
plt.title("Dispersão: Anúncios vs Cliques")
plt.xlabel("Quantidade de Anúncios")
plt.ylabel("Quantidade de Cliques")
plt.grid(True)
plt.show()

# --- ETAPA 3: PREPARAÇÃO DOS DADOS ---
X = df[['anuncios']]
y = df['cliques']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ETAPA 4: TREINAMENTO DO MODELO ---
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# --- ETAPA 5: VISUALIZAÇÃO E RESULTADOS ---
y_pred = modelo.predict(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Dados Reais')
plt.plot(X, modelo.predict(X), color='red', linewidth=2, label='Reta de Regressão')
plt.title("Modelo Final: Previsão de Cliques")
plt.xlabel("Anúncios")
plt.ylabel("Cliques")
plt.legend()
plt.show()

print(f"\nAcurácia do Modelo (R2 Score): {r2_score(y_test, y_pred):.2f}")