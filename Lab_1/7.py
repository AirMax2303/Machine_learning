import numpy as np

# Создание матрицы 3x3
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Нахождение детерминанта матрицы
determinant = np.linalg.det(matrix)

print("Матрица:")
print(matrix)
print("Детерминант матрицы:")
print(determinant)
