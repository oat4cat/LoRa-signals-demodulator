import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torchaudio
import json
import os
from model import CNN2D
import numpy as np

# Загружаем модель
model_path = "best_cnn2d_spec.pth"
device = "cpu"
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Преобразование в спектрограмму
def audio_to_spectrogram(waveform, n_fft=1024, hop_length=512):
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2
    )(waveform)
    return spectrogram.log2().clamp(min=-10)

# Функция для демодуляции
def demodulate(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)
    
    # Преобразуем в спектрограмму
    spec = audio_to_spectrogram(waveform)
    
    # Добавляем batch dimension и размерность канала
    # spec = spec.unsqueeze(0)  # [1, 1, freq_bins, time_steps]
    
    with torch.no_grad():
        output = model(spec)
    
    # Конвертируем тензоры в списки
    if isinstance(output, dict):
        result = {k: v.cpu().numpy().tolist() if torch.is_tensor(v) else v 
                 for k, v in output.items()}
    else:
        result = output.cpu().numpy().tolist()
    
    return result


# Функция выбора файла
def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        try:
            result = demodulate(file_path)

            save_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                     filetypes=[("JSON files", "*.json")])
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(result, f, indent=4)
                messagebox.showinfo("Успех", f"Файл успешно сохранен:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

# Создаем окно
root = tk.Tk()
root.title("LoRa Демодуляция WAV -> JSON")
root.geometry("400x200")

# Виджеты
label = tk.Label(root, text="Выберите WAV-файл для демодуляции", font=("Arial", 14))
label.pack(pady=20)

button = tk.Button(root, text="Выбрать файл", command=choose_file, font=("Arial", 12))
button.pack(pady=10)

root.mainloop()
