## Решение кейса "Сколтех - Наводнения" команды ThreeNearestNeighbours (MIPT, MISIS)

Тренировка:
    - Правим конфиг
    - Запускаем python main.py для обучения и валидации.
    - На валидации считается f1, iou и calculate_metrics.py
    - Модели сохраняются в папках run-0, run-1, ... в валидационной папке из конфига

Тест:
    - Указываем в run_test_data.py модели для тестирования
    - Запускаем python run_test_data.py
    - В папке с данными появляются папки предсказаний
    - Правим и запускаем python run_voting.py для получения финального результата

## Описание решения

1) Сначала мы провели анализ данных поэксперементировали с разлиными подходами - описание некоторых экспериментов и идей можно найти в [ноутбуке](edytor.ipynb). В частности мы загрузили данные о высоте местности и разметку водоемов с OSM.

2) Для сегментации мы пробовали различные архитектуры: Unet, DeepLab, трансформеры и другие. Главной проблемой было научить модель обобщаться на картинки с другой местностью. Для этого мы пытались использовать дополнительное обучение на данных датасета [WorldFloods](https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html) и различные техники нормализации и аугментации. Эксперименты проводили в Weights & Biases, модели сравнивали на метриках описанных в презентации.

3) Выбрали несколько наиболее удачных и слабо скореллированных моделей (обученных с различными признаками) и скомбинировали их результаты.

4) Проводили постобработку используя карту высот - распростряняли воду с более высоких мест в более низкие.

## Результаты

Мы валидировали модель на картинках, отсутствующих в обучающем датасете, и получили следующие максимальные результаты:

f1 предсказания воды: **0.865**

iou: **0.8936**

f1 по затопленным домам на картинке 9_2 (ее не было в обучающей части): **0.842**

Благодаря хорошей обобщающей способности модели на тесте мы получили еще больший скор: **0.937**

Пример работы:

![primer](example.jpg)

## Структура проекта

[download_data.sh](download_data.sh) - скрипт для загрузки обучающих и тестовых данных, скачивает их в папку dataset

[eda.ipynb](eda.ipynb) - ноутбук с графиками и анализом данных

Весь основной код обучения находится в папке src. Для обучения необходимо заполнить поля в [конфиге](baseline/config) и запустить [main.py](baseline/main.py). Для инференсаи получения предсказаний используется [run_test_data.py](baseline/run_test_data.py) - для получения предсказаний моделей, и [run_voting.py](baseline/run_voting.py) - для их блендинга.


[Ссылка](https://disk.yandex.ru/d/e2F_YR45dimlkw) на чекпонит модели resnet101, которая дала финальный скор
