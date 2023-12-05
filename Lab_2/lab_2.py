import pandas as pd

# Загрузка данных
df = pd.read_csv('math_students.csv')

print("Первые 10 строк:")
print(df.head(10))

print("Последние 10 строк:")
print(df.tail(10))

print("Информация о данных:")
print(df.describe())

print("Названия колонок:")
print(df.columns)

print("Есть ли в данных пропуски")
print(df.isnull().any().any())

print("Статистика по значениям признаков")
print(df.describe())

print("Подробная cтатистика по значениям признаков")
print(df.info())

print("Вывести более подробное описание значений признаков (количество непустых значений, типов столбцов и объема занимаемой памяти)")
print(df[df['age'].isnull()])


