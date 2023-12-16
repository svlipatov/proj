import pickle
import csv

# Файл инициализации модели
def init_model():
    # Загрузить модели из файла
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)

    # Считывание категорий
    file = open("cat.csv", "r")
    cat1 = list(csv.reader(file, delimiter=","))
    categorias = cat1[0]
    file.close()
    model.eval()
    return model, categorias