"""
Anàlisi i clustering dels ciclistes utilitzant KMeans.
"""
import os
import sys  # Importado para usar sys.exit
import logging
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes.

    Args:
        path (str): Ruta al dataset.

    Returns:
        pd.DataFrame: Dataset carregat.
    """
    try:
        dataframe = pd.read_csv(path)
        logging.info("S'ha carregat el dataset de %s", path)
        return dataframe
    except FileNotFoundError:
        logging.error("El fitxer no existeix: %s", path)
        return None

def exploratory_data_analysis(dataframe):
    """
    Exploració inicial del dataset.

    Args:
        dataframe (pd.DataFrame): Dataset.

    Returns:
        None
    """
    logging.info("Primeres files del dataset:\n%s", dataframe.head())
    logging.info("Descripció estadística del dataset:\n%s", dataframe.describe())
    logging.info("Informació del dataset:\n%s", dataframe.info())

def clean_data(dataframe):
    """
    Elimina les columnes no necessàries.

    Args:
        dataframe (pd.DataFrame): Dataset.

    Returns:
        pd.DataFrame: Dataset netejat.
    """
    return dataframe.drop(columns=["id", "t_total"], errors="ignore")

def extract_true_labels(dataframe):
    """
    Extreu les etiquetes reals dels ciclistes.

    Args:
        dataframe (pd.DataFrame): Dataset.

    Returns:
        np.ndarray: Etiquetes reals (labels).
    """
    labels_true = dataframe["category"].values  
    dataframe.drop(columns=["category"], inplace=True, errors="ignore")
    return labels_true

def visualitzar_pairplot(dataframe):
    """
    Genera un pairplot per analitzar les relacions entre atributs.

    Args:
        dataframe (pd.DataFrame): Dataset.

    Returns:
        None
    """
    sns.pairplot(dataframe)
    plt.savefig("./img/pairplot.png")
    logging.info("S'ha generat el pairplot i s'ha desat a ./img/pairplot.png")

def clustering_kmeans(data, n_clusters=4):
    """
    Aplica el model de clustering KMeans.

    Args:
        data (np.ndarray): Atributs (t_pujada, t_baixada).
        n_clusters (int): Nombre de clústers.

    Returns:
        KMeans: Model entrenat.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    logging.info("S'ha entrenat el model KMeans amb %d clusters.", n_clusters)
    return model

def associar_clusters_patrons(model, patterns):
    """
    Associa els clústers als patrons de comportament.

    Args:
        model (KMeans): Model entrenat.
        patterns (list): Llista de patrons.

    Returns:
        list: Patrons amb els labels assignats.
    """
    sorted_indices = model.cluster_centers_.sum(axis=1).argsort()
    for i, index in enumerate(sorted_indices):
        patterns[i]["label"] = index
    logging.info("S'han associat els clústers amb els patrons:\n%s", patterns)
    return patterns

def visualitzar_clusters(data, labels):
    """
    Genera un gràfic dels clústers.

    Args:
        data (np.ndarray): Dades.
        labels (np.ndarray): Etiquetes dels clústers.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", s=50)
    plt.xlabel("Temps de pujada (tp)")
    plt.ylabel("Temps de baixada (tb)")
    plt.title("Clusters dels ciclistes")
    plt.savefig("./img/clusters.png")
    logging.info("S'ha generat la visualització dels clusters i s'ha desat a ./img/clusters.png")

# ----------------------------------------------

if __name__ == "__main__":
    # Configuració del logging
    logging.basicConfig(level=logging.INFO)

    # Constants
    PATH_DATASET = '../P1/data/ciclistes.csv'

    # Crear directoris
    os.makedirs("./img", exist_ok=True)
    os.makedirs("./model", exist_ok=True)

    # Càrrega del dataset
    df = load_dataset(PATH_DATASET)
    if df is None:
        sys.exit("Error: No s'ha pogut carregar el dataset.")  # Usar sys.exit en lugar de exit

    # Exploració de dades
    exploratory_data_analysis(df)

    # Neteja del dataset
    df_clean = clean_data(df)

    # Extracció d'etiquetes reals
    labels_true = extract_true_labels(df_clean)  # Cambiado para usar "category"

    # Visualització del pairplot
    visualitzar_pairplot(df_clean)

    # Entrenament del model KMeans
    clustering_model = clustering_kmeans(df_clean[["t_pujada", "t_baixada"]].values)

    # Associar clústers als patrons
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]
    tipus = associar_clusters_patrons(clustering_model, tipus)

    # Guardar tipus_dict.pkl
    with open("./model/tipus_dict.pkl", "wb") as f:
        pickle.dump(tipus, f)
    logging.info("S'ha desat tipus_dict.pkl a ./model/")
