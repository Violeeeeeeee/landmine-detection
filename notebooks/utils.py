import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


def init_full_results_storage(filename):
    """
    Ініціалізує єдиний JSON-файл для всіх моделей проєкту.
    """
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            print(f"Файл {filename} знайдено. Завантажуємо існуючу структуру.")
            return json.load(f)
            
    # Базовий каркас для всіх конфігурацій
    base_structure = {
        "metadata": {
            "window_size": 32,
            "step": 17,
            "epochs": 50,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss_function": "MSELoss"
        },
        "results": {
            "1_channel": {
                "A1_x16": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []},
                "A2_x32": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []},
                "A3_x64": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []}
            },
            "3_channel": {
                "A1_x16": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []},
                "A2_x32": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []},
                "A3_x64": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []}
            },
            "5_channel": {
                "A1_x16": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []},
                "A2_x32": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []},
                "A3_x64": {"train_loss": [], "test_image_mse": [], "test_latent_mse": []}
            }
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(base_structure, f, indent=4)
    print(f"Створено новий глобальний файл {filename} для всіх 9 моделей.")
    
    return base_structure

def save_model_results(channel_type, model_key, train_loss, test_img_mse, test_lat_mse, filename):
    """
    Зберігає результати у відповідну секцію (наприклад: '1_channel', 'A1_x16').
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Перевірка на дурня
    if channel_type not in data["results"]:
        raise ValueError(f"Секції {channel_type} не існує! Використовуй 1_channel, 3_channel або 5_channel")
    if model_key not in data["results"][channel_type]:
        raise ValueError(f"Моделі {model_key} не існує в секції {channel_type}!")

    # Записуємо масиви
    data["results"][channel_type][model_key]["train_loss"] = train_loss
    data["results"][channel_type][model_key]["test_image_mse"] = test_img_mse
    data["results"][channel_type][model_key]["test_latent_mse"] = test_lat_mse
    
    # Зберігаємо на диск
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ Результати збережено: [{channel_type} -> {model_key}]")

# ==========================================
# УТИЛІТИ ДЛЯ ОЦІНКИ ТА ВІЗУАЛІЗАЦІЇ
# ==========================================

def evaluate_anomaly_detector(model, dataloader, device, criterion):
    """
    Проганяє тестовий датасет через модель і рахує Image MSE та Latent MSE.
    Повертає два списки з помилками.
    """
    model.eval()
    img_mse_list = []
    lat_mse_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            x_hat, z1, z2 = model(batch)
            
            img_mse_list.append(criterion(x_hat, batch).item())
            lat_mse_list.append(criterion(z1, z2).item())
            
    return img_mse_list, lat_mse_list

def plot_model_metrics(train_loss, img_mse, lat_mse, model_name, anomaly_win_start=19, anomaly_win_end=23):
    """
    Будує стандартний подвійний графік: падіння Loss та результати інференсу.
    """
    plt.figure(figsize=(14, 5))

    # 1. Графік тренування
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, color='blue', linewidth=2)
    plt.title(f"{model_name}: Тренувальний Loss")
    plt.xlabel("Епохи")
    plt.ylabel("MSE")
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Графік інференсу
    plt.subplot(1, 2, 2)
    x_axis = np.arange(1, len(lat_mse) + 1)
    
    # Визначаємо маркер залежно від моделі для консистентності
    marker_style = 'o' if 'A1' in model_name else ('s' if 'A2' in model_name else '^')
    color_style = 'red' if 'A1' in model_name else ('green' if 'A2' in model_name else 'blue')
    
    plt.plot(x_axis, lat_mse, marker=marker_style, color=color_style, label='Latent MSE ($Z_1 - Z_2$)')
    plt.plot(x_axis, img_mse, marker='x', color='orange', alpha=0.6, label='Image MSE')

    # Підсвітка зони
    plt.axvspan(anomaly_win_start, anomaly_win_end, color='gray', alpha=0.15, label='Фактична міна')

    plt.title(f"{model_name}: Інференс (Тестовий B-scan)")
    plt.xlabel("Номер патча (Вікно)")
    plt.ylabel("Рівень помилки")
    plt.xticks(np.arange(1, len(lat_mse) + 1, step=2))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_bscan_with_prediction(tensor_data, lat_mse, window_size=32, step=17, true_start=None, true_end=None, title="Тестовий B-scan"):
    """
    Малює теплову карту аномальності. Усі вікна підсвічуються червоним, 
    але інтенсивність (alpha) залежить від висоти помилки у цьому вікні.
    Ground Truth вимкнено за замовчуванням.
    """
    # Переводимо тензор у 2D масив NumPy
    if torch.is_tensor(tensor_data):
        bscan_2d = tensor_data.squeeze().cpu().numpy()
    else:
        bscan_2d = np.squeeze(tensor_data)
        
    plt.figure(figsize=(15, 6))
    img = plt.imshow(bscan_2d, aspect='auto', cmap='gray', vmin=-1, vmax=1)
    plt.grid(color='cyan', linestyle='--', linewidth=0.5, alpha=0.3)

    # 1. Нормалізація масиву помилок для розрахунку прозорості (Alpha)
    mse_array = np.array(lat_mse)
    min_err = np.min(mse_array)
    max_err = np.max(mse_array)
    
    # Захист від ділення на нуль (випадок "мертвої" мережі A3)
    if max_err > min_err:
        norm_mse = (mse_array - min_err) / (max_err - min_err)
    else:
        norm_mse = np.zeros_like(mse_array)

    # 2. Малюємо кожне вікно з відповідною інтенсивністю
    max_alpha = 0.5 # Максимальна непрозорість для найвищого спайку
    
    for i, val in enumerate(norm_mse):
        start = i * step
        end = start + window_size
        
        # Рахуємо альфу для поточного вікна
        alpha_val = val * max_alpha
        
        # Малюємо тільки якщо є хоч якась помітна помилка, щоб не перевантажувати рендер
        if alpha_val > 0.05:
            plt.axvspan(start, end, color='red', alpha=alpha_val, lw=0)

    # 3. Малюємо Ground Truth ТІЛЬКИ якщо передані координати
    if true_start is not None and true_end is not None:
        plt.axvspan(true_start, true_end, facecolor='none', edgecolor='green', 
                    linewidth=2, linestyle='--', label='Фактична міна (GT)')
        plt.legend(loc='upper right')

    plt.colorbar(img, label='Нормалізована амплітуда [-1, 1]')
    plt.title(title)
    plt.xlabel("Номер траси (Вісь X - просування радара)")
    plt.ylabel("Глибина (Вісь Z)")
    plt.tight_layout()
    plt.show()

def plot_reconstruction_sample(model, dataset, device, patch_idx, title_prefix=""):
    """
    Бере конкретний патч із датасету, проганяє через модель і малює 
    оригінал та те, що відновив декодер (x_hat).
    Адаптовано для багатоканальних (1D, 3D, 5D) даних.
    """
    model.eval()
    
    # Витягуємо патч і додаємо вимір батчу
    original_patch = dataset[patch_idx].unsqueeze(0).to(device)
    
    with torch.no_grad():
        x_hat, _, _ = model(original_patch)
        
    # Прибираємо вимір батчу
    orig_img = original_patch.squeeze(0).cpu().numpy()
    recon_img = x_hat.squeeze(0).cpu().numpy()
    
    # АДАПТАЦІЯ: Якщо каналів більше 1, беремо центральний зріз
    if orig_img.ndim == 3:
        center_c = orig_img.shape[0] // 2
        orig_img = orig_img[center_c]
        recon_img = recon_img[center_c]
    elif orig_img.ndim > 2:
        orig_img = np.squeeze(orig_img)
        recon_img = np.squeeze(recon_img)
        
    mse_error = np.mean((orig_img - recon_img) ** 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    
    im1 = axes[0].imshow(orig_img, aspect='auto', cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title("Оригінал (Вхід X, Центральний канал)")
    axes[0].set_ylabel("Глибина")
    axes[0].set_xlabel("Траси (Ширина 32)")
    
    im2 = axes[1].imshow(recon_img, aspect='auto', cmap='gray', vmin=-1, vmax=1)
    axes[1].set_title(rf"Реконструкція (Вихід $\hat{{X}}$)\nMSE: {mse_error:.5f}")
    axes[1].set_xlabel("Траси (Ширина 32)")
    
    plt.suptitle(f"{title_prefix} - Патч №{patch_idx+1}")
    plt.tight_layout()
    plt.show()

def plot_comparison_from_json(json_file, channel_type='1_channel', anomaly_start=19, anomaly_end=23):
    """
    Читає JSON-базу і будує зведений графік Latent MSE для всіх моделей 
    у межах заданої категорії каналів.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл {json_file} не знайдено.")
        return

    if channel_type not in data.get('results', {}):
        print(f"❌ Секцію {channel_type} не знайдено в JSON.")
        return
        
    models_data = data['results'][channel_type]
    
    plt.figure(figsize=(15, 7))
    
    # Хардкодимо стилі для стабільної візуалізації (щоб кольори не стрибали при перезапусках)
    styles = {
        'A1_x16': {'marker': 'o', 'color': 'red', 'label': 'A1 (x16)'},
        'A2_x32': {'marker': 's', 'color': 'green', 'label': 'A2 (x32)'},
        'A3_x64': {'marker': '^', 'color': 'blue', 'label': 'A3 (x64)'}
    }
    
    max_len = 0
    plotted_any = False
    
    for model_key, metrics in models_data.items():
        lat_mse = metrics.get('test_latent_mse', [])
        
        if not lat_mse:
            print(f"⚠️ Для моделі {model_key} немає даних test_latent_mse. Пропускаємо.")
            continue
            
        x_axis = np.arange(1, len(lat_mse) + 1)
        max_len = max(max_len, len(lat_mse))
        plotted_any = True
        
        # Беремо стиль або генеруємо дефолтний, якщо з'явиться нова модель
        style = styles.get(model_key, {'marker': 'x', 'color': 'black', 'label': model_key})
        
        plt.plot(x_axis, lat_mse, marker=style['marker'], color=style['color'], 
                 linewidth=2, alpha=0.8, label=style['label'])

    if plotted_any:    
        plt.title(f"Порівняння архітектур ({channel_type}): Latent Error $Z_1 - Z_2$")
        plt.xlabel("Номер патча (Вікно ковзає по профілю)")
        plt.ylabel("Рівень аномальності (MSE)")
        plt.xticks(np.arange(1, max_len + 1, step=1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left', fontsize='11')
        plt.tight_layout()
        plt.show()
    else:
        print("❌ Немає валідних даних для побудови графіка.")


def run_evaluation_pipeline(model, model_key, channel_type, test_loader, test_dataset, test_tensor, train_loss_history, device, criterion_mse, filename):
    """
    Комплексно проганяє інференс, зберігає метрики та будує всі необхідні графіки.
    """
    print(f"\n=== ІНФЕРЕНС ТА ВІЗУАЛІЗАЦІЯ: {model_key} ({channel_type}) ===")
    
    # 1. Проганяємо через модель
    img_mse, lat_mse = evaluate_anomaly_detector(model, test_loader, device, criterion_mse)
    print(f"Оброблено {len(lat_mse)} патчів.")
    
    # 2. Зберігаємо результати у JSON (переконайся, що функція save_model_results є у твоєму файлі)
    save_model_results(channel_type, model_key, train_loss_history, img_mse, lat_mse, filename)
    
    # 3. Графіки тренування та інференсу
    plot_model_metrics(train_loss_history, img_mse, lat_mse, model_name=f"{channel_type}_{model_key}")
    
    # 4. Теплова карта B-scan (Автоматично бере центральний зріз для 3D/5D)
    center_idx = test_tensor.shape[0] // 2 if test_tensor.ndim == 3 else 0
    central_bscan = test_tensor[center_idx] if test_tensor.ndim == 3 else test_tensor
    
    plot_bscan_with_prediction(
        tensor_data=central_bscan, 
        lat_mse=lat_mse, 
        title=f"{model_key}: Теплова карта на центральній лінії"
    )
    
    # 5. Приклад реконструкції чистого ґрунту (наприклад, 8-й патч, де немає міни)
    plot_reconstruction_sample(model, test_dataset, device, patch_idx=8, title_prefix=f"{model_key} (Чистий ґрунт)")