import sys
import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans
import pandas as pd

# Agregar la ruta absoluta del directorio P2 al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'P2')))

# Ahora puedes importar el archivo clustersciclistes.py desde P2
from clustersciclistes import load_dataset, clean_data, extract_true_labels, clustering_kmeans

# Configuración de logging
logging.basicConfig(level=logging.INFO)

# Ruta del dataset
PATH_DATASET = '../P1/data/ciclistes.csv'

# Cargar el dataset
df = load_dataset(PATH_DATASET)
if df is None:
    sys.exit("Error: No s'ha pogut carregar el dataset.")

# Limpiar el dataset
df_clean = clean_data(df)

# Extraer las etiquetas verdaderas
labels_true = extract_true_labels(df_clean)

# Configurar el experimento en MLflow
mlflow.set_experiment("K sklearn ciclistes")

# Realizar una variación del parámetro K desde K=2 hasta K=8
for k in range(2, 9):
    with mlflow.start_run():
        # Aplicar KMeans
        clustering_model = clustering_kmeans(df_clean[["t_pujada", "t_baixada"]].values, n_clusters=k)
        
        # Predecir las etiquetas
        labels_pred = clustering_model.predict(df_clean[["t_pujada", "t_baixada"]].values)
        
        # Calcular las métricas
        homogeneity = homogeneity_score(labels_true, labels_pred)
        completeness = completeness_score(labels_true, labels_pred)
        v_measure = v_measure_score(labels_true, labels_pred)
        
        # Log de los parámetros y métricas
        mlflow.log_param("K", k)
        mlflow.log_metric("homogeneity_score", homogeneity)
        mlflow.log_metric("completeness_score", completeness)
        mlflow.log_metric("v_measure_score", v_measure)
        
        logging.info(f"Run para K={k} completado con éxito. Métricas guardadas en MLflow.")

print('S\'han generat els runs.')
