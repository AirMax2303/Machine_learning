import numpy as np

# Создание первой матрицы 2x2
matrix1 = np.array([[1, 2], [3, 4]])

# Создание второй матрицы 2x3
matrix2 = np.array([[5, 6, 7], [8, 9, 10]])

# Нахождение произведения первой матрицы на транспонированную вторую матрицу

result = np.dot(matrix1, matrix2.T, out=None)

print("Первая матрица:")
print(matrix1)
print("Вторая матрица:")
print(matrix2)
print('T:')
print(matrix2.T)
print("Результат произведения первой матрицы на транспонированную вторую матрицу:")
print(result)
