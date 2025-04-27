import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.io import wavfile
from scipy import signal
from tqdm.auto import tqdm
import numpy as np

def plot_wav(data, sample_rate, title, prefix='fig', markers=None):
    """Визуализация WAV-файла с маркерами."""
    plt.figure(figsize=(15, 5))
    time = np.arange(len(data)) / sample_rate
    
    # Для двухканального отображаем оба канала
    if data.ndim == 2:
        plt.plot(time, data[:, 0], label='Канал 1', alpha=0.7)
        plt.plot(time, data[:, 1], label='Канал 2', alpha=0.7)
    else:
        plt.plot(time, data, label='Аудио')
    
    if markers:
        for mark in markers:
            plt.axvline(mark / sample_rate, color='r', linestyle='--', alpha=0.5)
    
    plt.title(title)
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(prefix + '.png')
    

def remove_silence(wav_data, sample_rate, threshold_db=0,
                   min_silence_duration=0.1, window_size=0.02, verbose=True):
    """
    Удаление тишины по энергетическому порогу
    
    Параметры:
        threshold_db - порог в децибелах относительно средней энергии
        min_silence_duration - минимальная длительность тишины для удаления (сек)
        window_size - размер окна для анализа (сек)
    """
    # 1. Преобразуем в моно и нормализуем
    if verbose:
        print("🔍 Подготовка данных...")
    
    if wav_data.ndim == 2:
        mono = np.mean(wav_data, axis=1)
    else:
        mono = wav_data.copy()
    
    mono = mono / np.max(np.abs(mono))
    
    # 2. Разбиваем на окна
    samples_per_window = int(window_size * sample_rate)
    num_windows = len(mono) // samples_per_window
    
    if verbose:
        print(f"📊 Анализ {num_windows} окон...")
    
    # 3. Рассчитываем энергию в каждом окне
    energies = []
    with tqdm(total=num_windows, desc="Расчёт энергии", disable=not verbose) as pbar:
        for i in range(num_windows):
            start = i * samples_per_window
            end = start + samples_per_window
            window = mono[start:end]
            energy = 10 * np.log10(np.mean(window**2) + 1e-10)  # в dB
            energies.append(energy)
            pbar.update(1)
    
    # 4. Определяем порог
    avg_energy = np.mean(energies)
    silence_threshold = avg_energy + threshold_db
    
    if verbose:
        print(f"⚡ Средняя энергия: {avg_energy:.1f} dB")
        print(f"🔇 Порог тишины: {silence_threshold:.1f} dB")
    
    # 5. Помечаем активные окна
    is_active = np.array(energies) > silence_threshold
    
    # 6. Удаляем короткие паузы (моргания)
    min_silence_windows = int(min_silence_duration / window_size)
    is_active_smoothed = np.convolve(
        is_active.astype(float),
        np.ones(min_silence_windows),
        mode='same'
    ) > 0.5
    
    # 7. Создаём маску для оригинальных сэмплов
    mask = np.repeat(is_active_smoothed, samples_per_window)
    mask = np.pad(mask, (0, len(mono) - len(mask)), mode='constant')
    
    # 8. Применяем маску (сохраняем оба канала если было 2 канала)
    cleaned_data = wav_data[mask[:len(wav_data)]]


    if verbose:
        print(f"✅ Готово! Новый размер: {len(cleaned_data)} сэмплов ({len(cleaned_data)/sample_rate:.2f} сек)")
        percent = 100*(1 - len(cleaned_data)/len(wav_data))
        print(f"Удалено {percent:.1f}% данных")
    
    return cleaned_data, percent


def process_files(wav_path, txt_path, prefix, output_dir='', silence_threshold=0.1, plot=True):
    # Создаём директории
    os.makedirs(os.path.join(output_dir, 'wav'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'txt'), exist_ok=True)
    
    # Читаем WAV и показываем исходный сигнал
    sample_rate, wav_data = wavfile.read(wav_path)
    print("Загрузили файл")
    # if plot:
    #    plot_wav(wav_data, sample_rate, 'Исходный WAV с тишиной', wav_path.split('.')[0])
    
    # Удаляем тишину
    wav_data, percent = remove_silence(wav_data, sample_rate)
    print("Удалили тишину")
    if percent > 95 or percent < 5:
        print("Некорректные пороги тишины")
        return
    if plot:
        plot_wav(wav_data, sample_rate, 'WAV после удаления тишины', wav_path.split('.')[0])
    
    total_samples = wav_data.shape[0]

    # Читаем JSON сигналы
    print("Читаем джсон")
    with open(txt_path, 'r') as f:
        signals = [json.loads(line) for line in f.readlines() if line.strip()]
    
    # Группируем по полю 'c'
    signal_groups = defaultdict(list)
    for signal in signals:
        signal_groups[signal['c']].append(signal)
    
    sorted_groups = [signal_groups[c] for c in sorted(signal_groups.keys())]
    
    # Классифицируем группы
    classified_groups = []
    for group in sorted_groups:
        short = [sig for sig in group if 'pc' in sig]
        long = [sig for sig in group if 'dr' in sig]
        classified_groups.append((short[0] if short else None, long[0] if long else None))
    
    # Определяем длины сигналов
    total_signals = len(classified_groups)
    N_short = sum(1 for group in classified_groups if group[0] is not None)
    N_long = sum(1 for group in classified_groups if group[1] is not None)
    
    K = 37/7  # Соотношение длин long/short (настройте под ваши данные)
    short_len = total_samples // (N_short + N_long * K)
    long_len = K * short_len
    
    print(f"Автоматически определены длины: short={short_len} samples ({short_len/sample_rate:.3f} сек), "
          f"long={long_len} samples ({long_len/sample_rate:.3f} сек)")

    # Разбиваем WAV и сохраняем сегменты
    current_sample = 0
    file_counter = 0
    markers = []

    for group in classified_groups:
        short_sig, long_sig = group
        segment_length = 0
        
        if short_sig and long_sig:
            segment_length = int(short_len + long_len)
        elif short_sig:
            segment_length = int(short_len)
        elif long_sig:
            segment_length = int(long_len)
        else:
            continue
        
        markers.extend([current_sample, current_sample + segment_length])
        
        # Сохраняем WAV сегмент
        wav_segment = wav_data[current_sample:current_sample+segment_length]
        wavfile.write(os.path.join(output_dir, 'wav', f'{prefix}_{file_counter:04d}.wav'), 
                     sample_rate, wav_segment)
        
        # Сохраняем TXT данные
        txt_data = []
        if short_sig:
            txt_data.extend([v for k, v in short_sig.items() if k != 'c' and isinstance(v, (int, float))])
        if long_sig:
            txt_data.extend([v for k, v in long_sig.items() if k != 'c' and isinstance(v, (int, float))])
        
        with open(os.path.join(output_dir, 'txt', f'{prefix}_{file_counter:04d}.txt'), 'w') as f:
            f.write(' '.join(map(str, txt_data)))
        
        current_sample += segment_length
        file_counter += 1
    print("Визуализация разбиения")
    # Визуализация разбиения
    if plot and markers:
        plot_wav(wav_data, sample_rate, 'Разметка сегментов', prefix=prefix, markers=markers)

if __name__ == '__main__':
    input_wav = 'out_stereo_8gb.wav'
    input_txt = 'decoded_8gb.txt'
    prefix = '8gb'
    process_files(input_wav, input_txt, prefix, silence_threshold=0.5)