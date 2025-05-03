def one_hot(label: int, num_classes: int = 10) -> list[int]:
    """Преобразование метки класса в вектор"""
    vector = [0] * num_classes
    vector[label] = 1
    return vector


def argmax(vec: list[float]) -> int:
    """Нахождение индекса максимального значения в векторе"""
    return vec.index(max(vec))
