"""
Модуль высокоскоростной предобработки капч ФНС (чеков) через OpenCV и NumPy.
- Удаление регулярной диагональной сетки белых точек (медианный фильтр 3x3).
- Расчет цвета фона и вычисление цветовой дельты (Color Distance Delta-E).
- Бинаризация по Otsu.
- Детекция и удаление сквозной горизонтальной линии перечеркивания.
- Вертикальное залечивание микроразрывов штрихов цифр.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image


def clean_dots_and_background(img_bgr: np.ndarray) -> np.ndarray:
    """
    Удаляет точечную сетку и фон, возвращая бинарную маску цифр и помех.
    Время выполнения: ~0.10 мс.
    """
    # 1. Подавление точечного шума (сетки белых точек)
    denoised = cv2.medianBlur(img_bgr, 3)

    # 2. Оценка истинного цвета фона по 4 углам изображения 10x10 px
    c1 = denoised[:10, :10]
    c2 = denoised[:10, -10:]
    c3 = denoised[-10:, :10]
    c4 = denoised[-10:, -10:]
    corners = np.concatenate([
        c1.reshape(-1, 3),
        c2.reshape(-1, 3),
        c3.reshape(-1, 3),
        c4.reshape(-1, 3)
    ])
    bg_color = np.median(corners, axis=0).astype(np.float32)

    # 3. Евклидово расстояние каждого пикселя от фона в цветовом пространстве
    diff = np.linalg.norm(denoised.astype(np.float32) - bg_color, axis=2)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 4. Адаптивный порог Otsu
    _, thresh = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def remove_strikethrough_line(thresh: np.ndarray) -> np.ndarray:
    """
    Удаляет сквозную горизонтальную линию и залечивает вертикальные разрывы.
    Время выполнения: ~0.04 мс.
    """
    # Детекция длинной непрерывной горизонтальной полосы (ширина >= 25 px)
    k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    line = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_line)

    # Линия всегда проходит в центральной зоне высоты y: 20..60
    line_roi = np.zeros_like(thresh)
    line_roi[20:60, :] = line[20:60, :]

    # Вычитание обнаруженной линии
    clean = cv2.subtract(thresh, line_roi)

    # Вертикальное закрытие для сращивания разрывов штрихов
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    repaired = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_v)
    return repaired


def prepare_crop_and_pad(clean: np.ndarray) -> Optional[np.ndarray]:
    """
    Кадрирует рабочую область цифр и добавляет белый защитный отступ.
    Возвращает инвертированное ч/б изображение (черные символы на белом фоне).
    """
    ys, xs = np.where(clean > 0)
    if len(xs) < 40:
        return None

    min_x = max(0, int(xs.min()) - 8)
    max_x = min(clean.shape[1], int(xs.max()) + 8)
    min_y = max(0, int(ys.min()) - 8)
    max_y = min(clean.shape[0], int(ys.max()) + 8)

    crop = clean[min_y:max_y, min_x:max_x]
    inv_crop = cv2.bitwise_not(crop)
    padded = cv2.copyMakeBorder(inv_crop, 10, 10, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return padded


def prepare_spaced_components(clean: np.ndarray) -> Optional[np.ndarray]:
    """
    Специализированный пространственный рефайнер для одинаковых цифр подряд (777, 811, 440).
    Разделяет связные компоненты и размещает их с гарантированным зазором 25 px.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean)
    valid_comps = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 30 and w >= 4 and 10 <= h <= 60:
            valid_comps.append((x, y, w, h, i))

    if not (2 <= len(valid_comps) <= 4):
        return None

    valid_comps = sorted(valid_comps, key=lambda c: c[0])
    total_w = sum(c[2] for c in valid_comps) + len(valid_comps) * 30 + 40
    canvas_w = max(200, total_w)
    canvas = np.zeros((80, canvas_w), dtype=np.uint8)

    cur_x = 20
    for x, y, w, h, idx in valid_comps:
        comp_mask = (labels[y:y+h, x:x+w] == idx).astype(np.uint8) * 255
        target_y = max(0, min(80 - h, (80 - h) // 2))
        canvas[target_y:target_y+h, cur_x:cur_x+w] = comp_mask
        cur_x += w + 25

    inv_canvas = cv2.bitwise_not(canvas)
    return inv_canvas
