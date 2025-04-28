# LoRa-signals-demodulator
Гибридная CNN-Transformer модель для демодуляции радиосигналов LoRa (преобразование .wav записей в структурированные данные)

# О проекте
Решение для декодирования LoRa-передач с использованием глубокого обучения.
Обрабатывает "сырые" RF-сигналы (2.4 МГц) с зонда и выдает полезную нагрузку в JSON.

* Гибридная архитектура: CNN + Transformer
* Адаптивная предобработка: Спектрограммы/FFT с автоматическим паддингом

![изображение](https://github.com/user-attachments/assets/9cbbab8a-af50-4765-807d-c27625dfb0a4)


# Для запуска проекта (app)

```
pip install -r requirements.txt

# Декодирование
python final.py
```



```
├── models/              # Готовые модели
│   ├── cnn2d_transformer.pt 
│   └── cnn2d.pt
├── data/                # Датасет
│   ├── wav/
│   ├── txt/
├── data_raw/            # Пример необработанных данных
│   ├── baseband.wav
├── app/                 # Приложение
├── notebook.ipynb       # Эксперименты с разными моделями
├── split_files.py       # скрипт для создания пар сигнал-метка из .wav файла
└── requirements.txt
```

