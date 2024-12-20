import unittest
import os
import pickle
import sys

# Agrega la ruta de P1 al PYTHONPATH
sys.path.insert(0, os.path.abspath('../../P1'))
sys.path.insert(0, os.path.abspath('../../P2'))

from generardataset import generar_dataset
from clustersciclistes import load_dataset, clean_data, extract_true_labels, clustering_kmeans


class TestGenerarDataset(unittest.TestCase):
    """
    Classe TestGenerarDataset
    """
    global mu_p_be
    global mu_p_me
    global mu_b_bb
    global mu_b_mb
    global sigma
    global dicc

    mu_p_be = 3240  # Mitjana temps pujada bons escaladors
    mu_p_me = 4268  # Mitjana temps pujada mals escaladors
    mu_b_bb = 1440  # Mitjana temps baixada bons baixadors
    mu_b_mb = 2160  # Mitjana temps baixada mals baixadors
    sigma = 240  # Desviació estàndard

    dicc = [
        {"name": "BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
        {"name": "BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
        {"name": "MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
        {"name": "MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
    ]

    def test_longituddataset(self):
        """
        Test de la longitud del dataset generat
        """
        arr = generar_dataset(200, 1, dicc)
        self.assertEqual(len(arr), 200)

    def test_valorsmitjatp(self):
        """
        Test del valor mitjà del temps de pujada (tp)
        """
        arr = generar_dataset(100, 1, dicc)
        arr_tp = arr["t_pujada"]
        tp_mig = sum(arr_tp) / len(arr_tp)
        self.assertLess(tp_mig, 3400)

    def test_valorsmitjatb(self):
        """
        Test del valor mitjà del temps de baixada (tb)
        """
        arr = generar_dataset(100, 1, dicc)
        arr_tb = arr["t_baixada"]
        tb_mig = sum(arr_tb) / len(arr_tb)
        self.assertGreater(tb_mig, 2000)


class TestClustersCiclistes(unittest.TestCase):
    """
    Classe TestClustersCiclistes
    """
    global ciclistes_data_clean
    global data_labels

    # Ajustar la ruta para acceder correctamente a la carpeta 'data' en P1
    path_dataset = os.path.join(os.path.abspath('../../P1'), 'data/ciclistes.csv')

    # Comprobar si el archivo existe antes de cargarlo
    if os.path.isfile(path_dataset):
        ciclistes_data = load_dataset(path_dataset)
        ciclistes_data_clean = clean_data(ciclistes_data)
        true_labels = extract_true_labels(ciclistes_data_clean)

        # Asegurarse de que el directorio 'model' exista antes de guardar el modelo
        model_dir = 'model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        clustering_model = clustering_kmeans(ciclistes_data_clean[["t_pujada", "t_baixada"]])
        with open(os.path.join(model_dir, 'clustering_model.pkl'), 'wb') as f:
            pickle.dump(clustering_model, f)
        data_labels = clustering_model.labels_
    else:
        ciclistes_data_clean = None
        data_labels = None
        print(f"El fitxer {path_dataset} no existeix.")

    def test_check_column(self):
        """
        Comprovem que una columna existeix al dataset
        """
        if ciclistes_data_clean is not None:
            self.assertIn("t_pujada", ciclistes_data_clean.columns)
        else:
            self.fail("El dataset no ha sido cargado correctamente.")

    def test_data_labels(self):
        """
        Comprovem que el nombre de data_labels és igual al nombre de registres
        """
        if data_labels is not None:
            self.assertEqual(len(data_labels), len(ciclistes_data_clean))
        else:
            self.fail("Los labels no se han generado debido a un error en el dataset.")

    def test_model_saved(self):
        """
        Comprovem que s'ha guardat el fitxer clustering_model.pkl
        """
        check_file = os.path.isfile('./model/clustering_model.pkl')
        self.assertTrue(check_file)


if __name__ == '__main__':
    unittest.main()
