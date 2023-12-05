import numpy as n
import random

X = [random.randint(0, 11) for x in range(4)]
Y = [random.randint(0, 11) for x in range(4)]
Z = [random.randint(0, 11) for x in range(4)]
G = [random.randint(0, 11) for x in range(4)]

matrix = n.array([X, Y, Z, G])
print(matrix)

res = []

for str in range(4):
    tmp = 1
    for row in range(4):
        tmp *= matrix[row, str]
    res.append(tmp)

print(f'Произведения элементов каждого столбца -> {res}')

SRED = sum(res)/4

print(f'Среднее значение -> {SRED}')

