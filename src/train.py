import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv("src/kaggle_comp_1/train.csv", index_col=[0])  #carregar dados
test_df = pd.read_csv("src/kaggle_comp_1/test.csv", index_col=[0])  

X = train_df.drop(columns=["class"])  
y = train_df["class"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) #divisão de treino e validacao


joblib.dump(X.columns.tolist(), "src/kaggle_comp_1/train_columns.pkl")  #salva colunas pós treinamento

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # normalizacao dos dados
X_val = scaler.transform(X_val)

knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train, y_train)              #treino

accuracy = knn.score(X_val, y_val)
print(f"acuracia: {accuracy:.4f}")       #avaliacao

joblib.dump(knn, "src/kaggle_comp_1/knn_model.pkl")
joblib.dump(scaler, "src/kaggle_comp_1/scaler.pkl")
print("pronto")        #salvar modelo

X_test = scaler.transform(test_df)
predictions = knn.predict(X_test)
