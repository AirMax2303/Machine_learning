import numpy as np

# Создание матрицы 5x5
matrix = np.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]])

# Вычисление суммы элементов по главной и побочной диагоналям матрицы
sum_main_diagonal = np.trace(matrix)
sum_secondary_diagonal = np.trace(np.fliplr(matrix))

print("Матрица:")
print(matrix)
print("Сумма элементов по главной диагонали:")
print(sum_main_diagonal)
print("Сумма элементов по побочной диагонали:")
print(sum_secondary_diagonal)
