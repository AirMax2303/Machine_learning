import pandas as pd

# Загрузка данных
df = pd.read_csv('math_students.csv')

# Вывод первых 10 строк
print("Первые 10 строк:")
print(df.head(10))

# Вывод последних 10 строк
print("Последние 10 строк:")
print(df.tail(10))

# Число объектов и их характеристик
print("Информация о данных:")
print(df.info())

# Названия всех колонок
print("nНазвания колонок:")
print(df.columns)
