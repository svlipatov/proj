---
title: Landmark Recognition
emoji: 🗺️
colorFrom: purple
colorTo: black
sdk: streamlit
sdk_version: 1.29.0
app_file: main.py
pinned: false
---
# Распознавание достопримечательностей

Данный проект предназначен для распознавания и вывода информации о достопримечательностях для улучшения опыта путешествий пользователей. По загруженным фотографиям приложение определит названия достопримечательностей, выведет короткие описания и отобразит их на карте.

![Alt text](images/demonstration.gif)

[Попробовать на HuggingFace](https://huggingface.co/spaces/molokhovdmitry/landmark_recognition)



## Установка и запуск

```
git clone https://github.com/svlipatov/proj
cd proj
pip install -r requirements.txt
streamlit run main.py
```

## Команда

**Молохов Д.А.** - ML-инженер по суммаризации текста.

**Таратута Е.Е.** - ML-инженер по подготовке и обработке датасета.

**Липатов С.В.** - ML-инженер по обучению модели распознавания изображений.

**Мальцев А.Ю.** - API-разработчик.

**Надеждин М.А.** - Разработчик интерфейса.



# proj
Smart city gid
В приложенных файлах:
1. check_photo_model_retrain.py - файл для переобучения последнего слоя модели googlenet обученной на датасете IMAGENET1K_V1.
2. pikle_model.pkl  - сохранненная в файл переобученная модель из п.1. Обучена на подготовленном датасете
3. check_photo.py - запуск распознования фотографии моделью. Возращает определнную категорию и вероятность в долях.
4. test_check_photo.py - Запуск распознования фото на файлах из тестовой выборки (в папке Test_photo)
5. check_photo_model_init.py инициализация модели. Модель поднимается из файла pickle_model.pkl, категории полнимаются из файла cat.csv
6. wikipedia_api.py - поиск на Википедия. Принимается список для поиска, возвращается список с результатами поиска
7. picturedownloader - парсинг фото для обучения модели по bing, через crawler
