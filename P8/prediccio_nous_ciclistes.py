import pickle
import pandas as pd
import os

# Ruta al modelo de clustering (teniendo en cuenta que está en P2/model)
model_path = os.path.join('..', 'P2', 'model', 'clustering_model.pkl')

# Cargar el modelo de clustering KMeans previamente entrenado
with open(model_path, 'rb') as f:
    clustering_model = pickle.load(f)

# Nuevos ciclistas con sus tiempos de pujada y baixada
nous_ciclistes = [
    [500, 3230, 1430, 4660],  # BEBB
    [501, 3300, 2120, 5420],  # BEMB
    [502, 4010, 1510, 5520],  # MEBB
    [503, 4350, 2200, 6550]   # MEMB
]

# Crear un DataFrame para los nuevos ciclistas con solo los tiempos de pujada y baixada
df_nous_ciclistes = pd.DataFrame(nous_ciclistes, columns=["id", "t_pujada", "t_baixada", "t_total"])

# Predecir los clusters utilizando el modelo de clustering entrenado
prediccions = clustering_model.predict(df_nous_ciclistes[["t_pujada", "t_baixada"]])

# Mostrar el resultado de las predicciones
df_nous_ciclistes["cluster"] = prediccions
print(df_nous_ciclistes[["id", "t_pujada", "t_baixada", "t_total", "cluster"]])
