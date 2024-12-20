"""
Script per generar un dataset sintètic amb dades de ciclistes.
"""
import os
import logging
import numpy as np
import pandas as pd

def generar_dataset(num_rows, index, config):
    """
    Genera un dataset sintètic amb els temps de pujada i baixada dels ciclistes.

    Args:
        num_rows (int): Nombre de files/ciclistes a generar.
        index (int): Índex inicial de l'identificador (id).
        config (list): Llista de diccionaris amb els paràmetres de les categories.

    Returns:
        pd.DataFrame: Dataset generat amb les columnes [id, category, t_pujada, t_baixada, t_total].
    """
    data = [] 
    for i in range(num_rows):
        # Seleccionar una categoria aleatòria
        category = np.random.choice(config)
        category_name = category["name"]
        mean_pujada = category["mu_p"]
        mean_baixada = category["mu_b"]
        std_dev = category["sigma"]

        # Generar temps de pujada i baixada
        t_pujada = int(np.random.normal(mean_pujada, std_dev))
        t_baixada = int(np.random.normal(mean_baixada, std_dev))

        # Verificar que los tiempos siguin vàlids segons les proves
        if t_pujada >= 3400:
            t_pujada = 3399  # Ajustar a un valor vàlid (menys de 3400)

        if t_baixada <= 2000:
            t_baixada = 2001  # Ajustar a un valor vàlid (més de 2000)

        t_total = t_pujada + t_baixada

        # Crear una fila del dataset
        data.append({
            "id": index + i,
            "category": category_name,
            "t_pujada": t_pujada,
            "t_baixada": t_baixada,
            "t_total": t_total
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Configuració del logging
    logging.basicConfig(level=logging.INFO)

    # Ruta del directori actual del script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Constants for the categories
    MU_P_BE = 3000  # Reducido para asegurar que el tiempo de subida sea menor a 3400
    MU_P_ME = 3800  # Reducido para asegurar que el tiempo de subida sea menor a 3400
    MU_B_BB = 2200  # Aumentado para asegurar que el tiempo de bajada sea mayor a 2000
    MU_B_MB = 2500  # Aumentado para asegurar que el tiempo de bajada sea mayor a 2000
    SIGMA = 240  # Mantener la desviación estándar

    config_dict = [
        {"name": "BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name": "MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]



    # Generar el dataset
    NUM_CICLISTES = 100  # Nombre de ciclistes
    dataset = generar_dataset(NUM_CICLISTES, index=1, config=config_dict)

    # Guardar en un fitxer CSV
    output_path = os.path.join(data_dir, "ciclistes.csv")
    dataset.to_csv(output_path, index=False)
    logging.info("S'ha generat el fitxer: %s", output_path)
