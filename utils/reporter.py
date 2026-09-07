"""
Модуль консольного мониторинга ресурсов, прогресс-бара и статистики.
"""

import os
import sys
import time
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None


def get_current_rss_mb() -> float:
    """Возвращает объем Resident Set Size памяти текущего процесса в МБ."""
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    return 0.0


class ProgressReporter:
    def __init__(self, total_items: int, quiet: bool = False):
        self.total_items = total_items
        self.quiet = quiet
        self.processed = 0
        self.three_digits_found = 0
        self.empty_frames_found = 0
        self.start_time = time.perf_counter()
        self.last_render_time = 0.0

    def update(self, count: int, three_digits: int, empty_frames: int):
        self.processed += count
        self.three_digits_found += three_digits
        self.empty_frames_found += empty_frames

        if self.quiet:
            return

        now = time.perf_counter()
        # Обновляем терминал не чаще 10 раз в секунду, чтобы не замедлять I/O
        if now - self.last_render_time < 0.1 and self.processed < self.total_items:
            return

        self.last_render_time = now
        elapsed = now - self.start_time
        fps = self.processed / elapsed if elapsed > 0 else 0.0
        eta = (self.total_items - self.processed) / fps if fps > 0 else 0.0
        pct = (self.processed / self.total_items) * 100.0 if self.total_items > 0 else 100.0
        ram_mb = get_current_rss_mb()

        # ASCII прогресс бар, совместимый с cp1251, cp866, utf-8
        bar_len = 20
        filled_len = int(bar_len * self.processed // max(1, self.total_items))
        bar = "#" * filled_len + "-" * (bar_len - filled_len)

        line = (
            f"\r[{bar}] {pct:5.1f}% ({self.processed}/{self.total_items}) | "
            f"Скорость: {fps:5.1f} кадр/с | ETA: {int(eta):02d}с | "
            f"RAM: {ram_mb:5.1f} МБ | 3-значных: {self.three_digits_found}"
        )
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except Exception:
            pass

    def finish(self):
        if not self.quiet:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass

    def print_final_summary(self, output_file: str):
        elapsed = time.perf_counter() - self.start_time
        fps = self.processed / elapsed if elapsed > 0 else 0.0
        ram_mb = get_current_rss_mb()
        extrapolated_10k_sec = (10000.0 / fps) if fps > 0 else 0.0

        print("=" * 65)
        print("ИТОГОВЫЙ ОТЧЕТ ОБРАБОТКИ ДАТАСЕТА")
        print("=" * 65)
        print(f"Всего обработано изображений:    {self.processed}")
        print(f"Обнаружено 3-значных чисел:      {self.three_digits_found} "
              f"({self.three_digits_found / max(1, self.processed):.1%})")
        print(f"Обнаружено пустых рамок чеков:   {self.empty_frames_found}")
        print(f"Затраченное время:               {elapsed:.2f} сек")
        print(f"Средняя скорость (Throughput):   {fps:.1f} изображений/сек")
        print(f"Пиковое потребление RAM:         {ram_mb:.1f} МБ (лимит: 500 МБ)")
        print(f"Расчетное время на 10 000 фото:  {extrapolated_10k_sec:.1f} сек "
              f"({extrapolated_10k_sec / 60.0:.2f} мин, лимит: 10 мин)")
        print(f"Результаты сохранены в CSV:      {output_file}")
        print("=" * 65)
