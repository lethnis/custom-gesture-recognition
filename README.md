# Распознавание жестов
Модель обучена распознавать следующие жесты: Thumb Up(Большой палец вверх), Peace(Мир),
Pointing Up(Указательный палец вверх), OK(Окей), Rock(Рок), Phone(Телефон), Love(Любовь), Fist(Кулак).  

Модель для нахождения рук и положения пальцев:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models

# Использование
1. Создать виртуальную среду и установить зависимости из файла `requirements.txt`.
2. Отредактировать файл `classes.txt`, добавив в него свои желаемые жесты.
3. Выполнить `python collect_images.py`. Программа соберёт по 500 изображений для каждого класса с вебкамеры. Можно использовать одну или две руки.
4. Выполнить `python create_dataset.py`. Программа проанализирует изображения и создаст из них файл с классами и координатами точек рук (по 21 точке для каждой руки).
5. Выполнить `python train_classifier.py`. Программа обучит классификатор, модель Random Forest.
6. Выполнить `python inference_classifier.py`. Программа запустить классификатор в режиме реального времени, пример на видео ниже.

# Пример обученной модели
https://github.com/lethnis/custom-gesture-recognition/assets/88483002/fb941fdf-e214-47bb-80aa-39055795b4cf
