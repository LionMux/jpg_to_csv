"""
Высокоскоростной движок нейросетевого OCR для капч строго из цифр [0-9].
Использует легковесный ONNX граф с маскированием логитов (Logits Masking)
и 3-стадийный каскадный рефайнер:
1. Стандартная фильтрация линии + вертикальное залечивание.
2. Сохранение линии (для ультратонких шрифтов, где вычитание линии съедает штрих).
3. Пространственное разделение с гарантированным гэпом (для слитных повторов 777, 811, 440).

Потребление RAM: ~70-90 МБ. Скорость: 5-8 мс/кадр.
"""

from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import cv2
import ddddocr

from recognizer.preprocessor import (
    clean_dots_and_background,
    remove_strikethrough_line,
    prepare_crop_and_pad,
    prepare_spaced_components,
)
from recognizer.empty_detector import is_empty_frame


class DigitOCREngine:
    def __init__(self):
        """Инициализация OCR движка."""
        self.ocr = ddddocr.DdddOcr(show_ad=False)
        self.engine = self.ocr.ocr_engine
        self.session = self.engine.session
        self.charset = self.engine.charset_manager.get_charset()

        # Поиск индексов цифр и CTC-blank (0) в словаре
        char_to_idx = {ch: i for i, ch in enumerate(self.charset)}
        digit_chars = "0123456789"
        self.digit_indices = [0] + [char_to_idx[c] for c in digit_chars if c in char_to_idx]

        # Вектор маски: разрешены только blank и цифры
        self.mask = np.full(len(self.charset), -1e9, dtype=np.float32)
        for idx in self.digit_indices:
            self.mask[idx] = 0.0

        self.input_name = self.session.get_inputs()[0].name

    def _ctc_decode(self, indices: np.ndarray) -> List[int]:
        """CTC декодер: удаляет непрерывные повторы и blank (0)."""
        res = []
        prev = 0
        for idx in indices:
            if idx != 0 and idx != prev:
                res.append(int(idx))
            prev = idx
        return res

    def _infer_image(self, inv_img: np.ndarray) -> str:
        """Инференс предобработанного ч/б изображения через ONNX модель."""
        pil_img = Image.fromarray(inv_img)
        processed = self.engine._preprocess_image(pil_img, False)
        raw_out = self.session.run(None, {self.input_name: processed})[0]

        # Наложение маски допустимых классов
        masked_out = raw_out + self.mask

        if raw_out.shape[1] == 1:
            pred_idx = np.argmax(masked_out[:, 0, :], axis=1)
        else:
            pred_idx = np.argmax(masked_out[0, :, :], axis=1)

        decoded = self._ctc_decode(pred_idx)
        return "".join([self.charset[i] for i in decoded])

    def recognize(self, img_bgr: np.ndarray) -> Tuple[str, str]:
        """
        Полный каскадный цикл решения капчи.
        Возвращает: (распознанные цифры, статус)
        """
        if img_bgr is None or img_bgr.size == 0:
            return "", "ERROR_EMPTY_IMAGE"

        # 1. Подавление шума и вычисление фона
        thresh = clean_dots_and_background(img_bgr)

        # 2. Проверка на пустые рамки ввода чека
        if is_empty_frame(thresh):
            return "", "EMPTY_FRAME"

        # 3. Стратегия А: удаление линии и залечивание
        clean = remove_strikethrough_line(thresh)
        padded = prepare_crop_and_pad(clean)

        pred = ""
        if padded is not None:
            pred = self._infer_image(padded)
            if len(pred) == 3:
                return pred, "OK"

        # 4. Стратегия Б: сохранение линии (для тонких штрихов)
        padded_raw = prepare_crop_and_pad(thresh)
        if padded_raw is not None:
            pred_raw = self._infer_image(padded_raw)
            if len(pred_raw) == 3:
                return pred_raw, "OK_RAW_LINE"

        # 5. Стратегия В: пространственный рефайнер для слитных повторов (777, 811)
        spaced = prepare_spaced_components(clean)
        if spaced is not None:
            pred_spaced = self._infer_image(spaced)
            if len(pred_spaced) == 3:
                return pred_spaced, "OK_SPACED"

        # Возвращаем лучший частичный результат
        candidates = [p for p in [pred, pred_raw if padded_raw is not None else ""] if p]
        best_pred = max(candidates, key=len) if candidates else ""

        if len(best_pred) == 3:
            return best_pred, "OK"
        elif len(best_pred) > 0:
            return best_pred, "PARTIAL"
        else:
            return "", "EMPTY"
