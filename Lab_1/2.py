import numpy as n

arr = n.array([3, 2, 1, 1, -1, -2, -3, 1, 1, 2, 1], int)
res = sum([x for x in arr if x > 0])
print(f'Сумма положительных элементов вектора = {res}')
