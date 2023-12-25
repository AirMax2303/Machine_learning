import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic_data = pd.read_csv('titanic.csv')

# 0. Первичный анализ данных
print(titanic_data.info())
print(titanic_data.describe())

print(titanic_data.isnull().sum())

titanic_data = titanic_data.dropna()

# 1. Построить график, показывающий количество выживших и погибших пассажиров в зависимости от порта посадки (признак Embarked).
sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Выжившие и погибшие пассажиры в зависиомти от порта посадки')
plt.show()

# 2. Исследовать распределение цен на билеты среди пассажиров в каждом классе каюты (признак Pclass), используя гистограммы.
plt.figure(figsize=(12, 6))
sns.histplot(x='Fare', hue='Pclass', data=titanic_data, bins=30, kde=True)
plt.title('Исследование цен на билеты в каждом классе каюты')
plt.show()

# 3. Проверить, есть ли зависимость между классом каюты (признак Pclass) и выживаемостью пассажиров. Построить столбчатую диаграмму, где по горизонтальной оси будут классы каюты, а по вертикальной - процент выживших и процент невыживших в каждом классе.
survival_by_class = titanic_data.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack()
survival_by_class.plot(kind='bar', stacked=True)
plt.title('Зависимость выживаемости от класса каюты')
plt.xlabel('Класс каюты')
plt.ylabel('% выживаемости')
plt.show()

# 4. Изучить корреляцию между возрастом пассажиров и стоимостью их билетов. Построить точечную диаграмму, где по горизонтальной оси будет возраст, а по вертикальной - стоимость билета. Разделить точки на группы по классу каюты и выделить их разными цветами.
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Age', y='Fare', hue='Pclass', data=titanic_data)
plt.xlabel('Возраст')
plt.ylabel('Стоимость билетов')
plt.title('Корреляция между возрастом пассажиров и стоимостью их билетов')
plt.show()

# 5. Сравнить распределение выживаемости пассажиров с разным количеством родственников на борту (признакы SibSp - число братьев/сестер/супругов и Parch - число родителей/детей). Построить график, где по горизонтальной оси будет количество родственников, а по вертикальной - процент выживших и невыживших пассажиров.
relatives_data = titanic_data[['SibSp', 'Parch', 'Survived']]
survival_by_relatives = relatives_data.groupby(['SibSp', 'Parch'])['Survived'].value_counts(normalize=True).unstack()
survival_by_relatives.plot(kind='bar', stacked=True)
plt.title('Распределение выживаемости пассажиров с разным количеством родственников на борту')
plt.xlabel('Количество родственников')
plt.ylabel('% выживаемости')
plt.show()

# 6. Определить соотношение мужчин и женщин среди пассажиров разных классов каюты. Построить круговую диаграмму, где каждый сектор будет отображать процентное соотношение мужчин и женщин в каждом классе.
gender_by_class = titanic_data.groupby('Pclass')['Sex'].value_counts().unstack()
gender_by_class.plot(kind='pie', subplots=True,  autopct='%1.1f%%', figsize=(15, 5))
plt.title('Соотношение мужчин и женщин среди пассажиров разных классов каюты')
plt.show()
