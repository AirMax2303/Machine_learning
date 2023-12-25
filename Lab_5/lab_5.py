import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Загрузка датасета Iris
iris_data = load_iris()
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target

# 2. Изучение структуры и содержания данных
print(iris_df.info())
print(iris_df.head())

# 3. Предварительный анализ данных
# Проверка наличия пропущенных значений
print(iris_df.isnull().sum())

# Корреляция между переменными
correlation_matrix = iris_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# 4. Анализ выбросов и удаление, если они есть (в данном случае, удалять не нужно)

# 5. Разделение данных на обучающую и тестовую выборки
X = iris_df.drop('target', axis=1)
y = iris_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Обучение модели логистической регрессии
logreg_model = LogisticRegression(max_iter=1000)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg_model.fit(X_train, y_train)

# 7. Оценка качества модели на тестовой выборке
y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

# 8. Визуализация результатов предсказания
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_pred)
plt.title('Предсказанные значения')
plt.show()

# 9. Кросс-валидация модели
cross_val_scores = cross_val_score(logreg_model, X, y, cv=5, scoring='accuracy')
print(f'Cross-validated Accuracy: {cross_val_scores.mean()}')

# 10. Использование модели RandomForestClassifier и поиск оптимальных параметров
rf_model = RandomForestClassifier()
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Обучение модели с оптимальными параметрами
rf_model = RandomForestClassifier(**best_params)
rf_model.fit(X_train, y_train)

# Оценка качества модели RandomForestClassifier
y_rf_pred = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_rf_pred)
precision_rf = precision_score(y_test, y_rf_pred, average='weighted')
recall_rf = recall_score(y_test, y_rf_pred, average='weighted')
f1_rf = f1_score(y_test, y_rf_pred, average='weighted')
conf_matrix_rf = confusion_matrix(y_test, y_rf_pred)

print(f'RandomForestClassifier - Accuracy: {accuracy_rf}')
print(f'RandomForestClassifier - Precision: {precision_rf}')
print(f'RandomForestClassifier - Recall: {recall_rf}')
print(f'RandomForestClassifier - F1 Score: {f1_rf}')
print(f'RandomForestClassifier - Confusion Matrix:\n{conf_matrix_rf}')

# 11. Анализ важности признаков для классификации и построение графика их влияния
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Важность признаков для классификации')
plt.show()
