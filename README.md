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
# 1. Клонируем репозиторий в sparse-режиме (только нужные файлы)
git clone --filter=blob:none --sparse https://github.com/oat4cat/LoRa-signals-demodulator
cd LoRa-signals-demodulator
git sparse-checkout init --cone
git sparse-checkout set app/

# 3. Создаем виртуальное окружение (Python 3.8+) и активируем его
python -m venv venv           # создает папку venv с изолированным Python
# source venv/bin/activate      # Linux/Mac
.\venv\Scripts\activate       # Windows (Cmd/PowerShell)


# 3. Устанавливаем зависимости и запускаем приложение
pip install -r requirements.txt
python app/final.py
```

# Структура репозитория

```
├── models/              # Готовые модели
│   ├── cnn2d_transformer.pt 
│   └── cnn2d.pt
├── data/                # Датасет
│   ├── wav/
│   └── txt/
├── data_raw/            # Пример необработанных данных
│   ├── baseband.wav
│   └── decoded.txt
├── app/                 # Приложение
├── notebook.ipynb       # Эксперименты с разными моделями
├── split_files.py       # скрипт для создания пар сигнал-метка из .wav файла
└── requirements.txt
```

