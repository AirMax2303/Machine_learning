import numpy as np

# Создание матрицы 2x3
matrix1 = np.array([[1, 2, 3], [4, 5, 6]])

# Создание матрицы 3x2
matrix2 = np.array([[7, 8], [9, 10], [11, 12]])

# Умножение матриц
result = np.dot(matrix1, matrix2)

print("Матрица 1:")
print(matrix1)
print("Матрица 2:")
print(matrix2)
print("Результат умножения матриц:")
print(result)
