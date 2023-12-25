import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Загрузка датасета из файла 'housing.csv'
housing_df = pd.read_csv('housing.csv', header=None, delim_whitespace=True)

# Назначение подходящих имен столбцам
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
housing_df.columns = column_names

# 2. Изучение структуры и содержание данных
print(housing_df.info())
print(housing_df.describe())

# 3. Предварительный анализ данных
# Проверка наличия пропущенных значений
print(housing_df.isnull().sum())

# Проверка выбросов и корреляции между переменными
sns.boxplot(x=housing_df['MEDV'])
plt.show()

correlation_matrix = housing_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# 4. Сравнение средних цен на недвижимость в различных районах
avg_prices_by_zone = housing_df.groupby('ZN')['MEDV'].mean()
print(avg_prices_by_zone)

# 5. Исследование влияния различных факторов на цену недвижимости
# Пример: корреляция между уровнем образования и ценой недвижимости
sns.scatterplot(x=housing_df['NOX'], y=housing_df['MEDV'])
plt.xlabel('NOX (нитрооксиды)')
plt.ylabel('Цена недвижимости')
plt.show()

# 6. Разделение данных на обучающую и тестовую выборки
X = housing_df.drop('MEDV', axis=1)
y = housing_df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Обучение модели линейной регрессии
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# 8. Оценка качества модели на тестовой выборке
y_pred = linear_reg_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')

# 9. Улучшение модели с использованием регуляризации (Lasso)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_y_pred = lasso_model.predict(X_test)

# 10. Визуализация результатов предсказания
plt.scatter(y_test, y_pred, label='Линейная регрессия')
plt.scatter(y_test, lasso_y_pred, label='Lasso')
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.legend()
plt.show()

# 11. Кросс-валидация модели
cross_val_scores = cross_val_score(linear_reg_model, X, y, scoring='neg_mean_squared_error', cv=5)
cross_val_rmse = np.sqrt(-cross_val_scores)

print(f'Cross-validated RMSE: {cross_val_rmse}')
