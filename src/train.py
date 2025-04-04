import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados corretamente
train_df = pd.read_csv('train.csv', index_col=[0])
test_df = pd.read_csv('test.csv', index_col=[0])

# Corrigir erro de digitação na leitura do CSV
# Removido "train_df = pd.read.csv('train.csv', index_col=[0])", pois já foi carregado antes

# Separar as features (X) e o target (y)
X = train_df.drop(columns=["Target"])  # Correção: "X" em vez de "x"
y = train_df["Target"]

# Dividir os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_val)

# Calcular a acurácia
accuracy = accuracy_score(y_val, y_pred)

# Corrigir erro de formatação na impressão
print(f"Acurácia: {accuracy:.4f}")