import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

boston_data = pd.read_csv('housing.csv')

#2: Изучите структуру и содержание данных.
print(boston_data.info())
print(boston_data.describe())

#3: Провести предварительный анализ данных, включая проверку наличия пропущенных значений, выбросов и корреляции между переменными.
print(boston_data.isnull().sum())

plt.figure(figsize=(15, 8))
sns.boxplot(data=boston_data, orient="h")
plt.title('Подробный план особенностей жилья в Бостоне')
plt.show()



