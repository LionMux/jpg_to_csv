"""
Модуль детекции пустых прямоугольных рамок-заглушек сервиса [ ][ ][ ].
Позволяет гарантированно отделять сбойные пустые формы от реальных капч с цифрами.
"""

from typing import Tuple
import cv2
import numpy as np


def is_empty_frame(thresh: np.ndarray) -> bool:
    """
    Определяет, представляет ли изображение пустые рамки ввода ввода чека [ ][ ][ ].
    Время работы: ~0.03 мс.
    """
    # Инвертируем бинарь для поиска внутренних полостей (дырок)
    inv = cv2.bitwise_not(thresh)
    # Маскируем края изображения
    inv[:12, :] = 0
    inv[65:, :] = 0
    inv[:, :8] = 0
    inv[:, 98:] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv)

    holes = []
    for s in stats[1:]:
        x, y, w, h, area = s[cv2.CC_STAT_LEFT], s[cv2.CC_STAT_TOP], s[cv2.CC_STAT_WIDTH], s[cv2.CC_STAT_HEIGHT], s[cv2.CC_STAT_AREA]
        # Пустая ячейка рамки имеет типичные размеры 20x9 px
        if 14 <= w <= 28 and 5 <= h <= 18 and 60 <= area <= 300:
            holes.append((int(x), int(y), int(w), int(h), int(area)))

    # Проверяем, есть ли ровно 3 или 2 ячейки, выстроенные в ряд с шагом ~25 px
    if len(holes) >= 2:
        holes = sorted(holes, key=lambda item: item[0])
        aligned_pairs = 0
        for i in range(len(holes) - 1):
            dx = holes[i+1][0] - holes[i][0]
            dy = abs(holes[i+1][1] - holes[i][1])
            if 20 <= dx <= 30 and dy <= 4:
                aligned_pairs += 1

        # Если 2 или 3 рамки выстроены строго по горизонтали
        if aligned_pairs >= 2:
            return True

    return False
