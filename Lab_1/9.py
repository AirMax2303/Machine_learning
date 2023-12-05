import numpy as np

# Создание вектора размера 15
vector = np.array([-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15])

# Подсчет количества отрицательных элементов вектора
num_negatives = np.sum(vector < 0)

print("Исходный вектор:")
print(vector)
print("Количество отрицательных элементов вектора:")
print(num_negatives)
