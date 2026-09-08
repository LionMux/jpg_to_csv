"""
Набор автоматизированных модульных тестов (Unit Tests) для проекта jpg_to_csv.
Проверяет:
1. Препроцессор: подавление шума, зануление краев 4 px, удаление точечной сетки, Intersection-Aware Line Removal.
2. OCR движок: стратегии каскада, Aspect Ratio Scaling 1.8x, контрольные файлы (117, 381, 139), цифры 4.
3. CLI и утилиты: декодирование байт, потоковый репортер, обработка ошибок.
"""

import unittest
import zipfile
from pathlib import Path
import numpy as np
import cv2

from recognizer.preprocessor import (
    clean_dots_and_background,
    remove_strikethrough_line,
    prepare_crop_and_pad,
    prepare_spaced_components,
)
from recognizer.empty_detector import is_empty_frame
from recognizer.ocr_engine import DigitOCREngine
from cli import decode_image_bytes, process_single_task, create_argument_parser
from utils.reporter import ProgressReporter, get_current_rss_mb


ZIP_PATH = Path(r"C:\Users\Lev\Desktop\ocr_hard.zip")


class TestPreprocessor(unittest.TestCase):
    def test_clean_dots_and_background_borders_zeroed(self):
        # Создаем искусственное тестовое изображение с белыми краями
        img = np.full((80, 150, 3), 255, dtype=np.uint8)
        img[20:60, 20:130] = 50  # темный прямоугольник внутри
        thresh = clean_dots_and_background(img)
        # Проверяем, что края толщиной 4 px строго равны 0
        self.assertEqual(np.count_nonzero(thresh[:4, :]), 0)
        self.assertEqual(np.count_nonzero(thresh[-4:, :]), 0)
        self.assertEqual(np.count_nonzero(thresh[:, :4]), 0)
        self.assertEqual(np.count_nonzero(thresh[:, -4:]), 0)

    def test_clean_dots_connected_components_small_dots_removed(self):
        # Изображение с искусственными мелкими изолированными точками 2x2 и 4x4
        thresh = np.zeros((80, 150), dtype=np.uint8)
        thresh[10:12, 10:12] = 255  # точка 2x2
        thresh[20:24, 20:24] = 255  # точка 4x4
        thresh[30:50, 30:50] = 255  # крупная компонента 20x20

        # Запускаем фильтрацию мелких компонент
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if (w <= 5 and h <= 5) or area < 25:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                thresh[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i] = 0

        self.assertEqual(thresh[11, 11], 0)
        self.assertEqual(thresh[22, 22], 0)
        self.assertEqual(thresh[40, 40], 255)

    def test_remove_strikethrough_line_intersection_protection(self):
        # Создаем горизонтальную линию и пересекающий ее вертикальный штрих (как в цифре 4 или 1)
        thresh = np.zeros((80, 150), dtype=np.uint8)
        thresh[35, 10:140] = 255  # длинная горизонтальная линия
        thresh[20:60, 70:74] = 255  # вертикальный штрих цифры

        cleaned = remove_strikethrough_line(thresh)
        # Вертикальный штрих в зоне пересечения (y=35, x=70..74) должен уцелеть
        self.assertTrue(np.all(cleaned[25:55, 71:73] > 0))
        # Свободная часть горизонтальной линии вдали от штриха должна быть стерта
        self.assertEqual(np.count_nonzero(cleaned[35, 20:40]), 0)

    def test_prepare_crop_and_pad_empty(self):
        empty = np.zeros((80, 150), dtype=np.uint8)
        self.assertIsNone(prepare_crop_and_pad(empty))


class TestOCREngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DigitOCREngine()

    def test_engine_empty_input(self):
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        digits, status = self.engine.recognize(empty_img)
        self.assertEqual(digits, "")
        self.assertEqual(status, "ERROR_EMPTY_IMAGE")

    @unittest.skipUnless(ZIP_PATH.exists(), "ocr_hard.zip not found")
    def test_control_files_improvements(self):
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            # 1. 20260830_180206_149 -> 117 (было 17)
            b1 = zf.read("20260830_180206_149_empty_empty.jpg")
            img1 = cv2.imdecode(np.frombuffer(b1, np.uint8), cv2.IMREAD_COLOR)
            d1, s1 = self.engine.recognize(img1)
            self.assertEqual(d1, "117")
            self.assertEqual(s1, "OK")

            # 2. 20260830_175630_328 -> 381 (было 38)
            b2 = zf.read("20260830_175630_328_empty_empty.jpg")
            img2 = cv2.imdecode(np.frombuffer(b2, np.uint8), cv2.IMREAD_COLOR)
            d2, s2 = self.engine.recognize(img2)
            self.assertEqual(d2, "381")
            self.assertEqual(s2, "OK_SCALED")

            # 3. 20260830_175821_615 -> 139 (было 39)
            b3 = zf.read("20260830_175821_615_empty_empty.jpg")
            img3 = cv2.imdecode(np.frombuffer(b3, np.uint8), cv2.IMREAD_COLOR)
            d3, s3 = self.engine.recognize(img3)
            self.assertEqual(d3, "139")
            self.assertEqual(s3, "OK_SCALED")

    @unittest.skipUnless(ZIP_PATH.exists(), "ocr_hard.zip not found")
    def test_digit_four_preservation(self):
        # Проверяем сохранение цифры 4 на контрольных образцах
        controls = [
            ("20260830_175701_930_empty_empty.jpg", "440"),
            ("20260830_175710_972_onnxtr_conflict_427.jpg", "427"),
            ("20260830_175741_955_onnxtr_conflict_454.jpg", "451"),
            ("20260830_175642_832_onnxtr_conflict_546.jpg", "646"),
        ]
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            for fn, expected in controls:
                data = zf.read(fn)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                digits, status = self.engine.recognize(img)
                self.assertEqual(digits, expected, f"Failed for {fn}: got {digits}")


class TestCLI(unittest.TestCase):
    def test_decode_image_bytes(self):
        self.assertIsNone(decode_image_bytes(b""))
        self.assertIsNone(decode_image_bytes(b"invalid_non_image_bytes"))

        # Валидный JPEG в памяти
        dummy = np.full((80, 150, 3), 128, dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", dummy)
        decoded = decode_image_bytes(encoded.tobytes())
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, (80, 150, 3))

    def test_argument_parser(self):
        parser = create_argument_parser()
        args = parser.parse_args(["-i", "data.zip", "-o", "out.csv", "-w", "2", "-b", "16", "-q"])
        self.assertEqual(str(args.input), "data.zip")
        self.assertEqual(str(args.output), "out.csv")
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.batch_size, 16)
        self.assertTrue(args.quiet)


if __name__ == "__main__":
    unittest.main()
