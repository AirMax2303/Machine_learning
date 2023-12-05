import numpy as n
import random

X = [random.randint(0, 11) for x in range(3)]
Y = [random.randint(0, 11) for x in range(3)]
Z = [random.randint(0, 11) for x in range(3)]

matrix = n.array([X, Y, Z])

SUM = [sum(x) for x in matrix]
SRED = sum(SUM)/3

print(matrix)
print(f'Суммы значений каждой строки-> {SUM}')
print(f'Среднее значение сумм строк-> {SRED}')
