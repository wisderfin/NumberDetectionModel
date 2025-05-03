import csv
from pathlib import Path

class CSVDataset:
    def __init__(self, path: Path, maximum: int = float('+inf')) -> None:
        """Загрузка датасета и приведенеие его к номальному ввиду"""
        self.path: Path = path
        self.labels: list[str] = []
        self.images: list[list[int]] = []
        self.maximum = maximum

        with open(self.path, encoding='utf-8') as f:
            next(f)
            for i, row in enumerate(csv.reader(f)):
                if i >= self.maximum:
                        break
                self.labels.append(int(row[0]))
                self.images.append([int(px) / 255 for px in row])

    def __len__(self):
        return len(self.labels)
