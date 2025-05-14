import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
import time

data = pd.read_csv('diabetes.csv')# Читаем данные из файла diabetes.csv

zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']# Создаем список признаков
for feature in zero_features: # Итерируемся по списку признаков
    data[feature] = data[feature].replace(0, np.nan) # Заменяем все нулевые значения в текущем признаке
    data[feature].fillna(data[feature].mean(), inplace=True) # Заменяем все NaN значения в текущем признаке на среднее значение этого признака

X = data.drop('Outcome', axis=1) # Создаем DataFrame X, содержащий все признаки
y = data['Outcome'] # Создаем y, содержащий целевую переменную 'Outcome'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Разделяем данные на обучающую (70%) и тестовую (30%) выборки

# Исследование зависимости качества от глубины деревьев
depth_values = range(1, 21) # Создаем список значений глубины дерева для исследования
f1_scores_depth = [] # Создаем пустой список для хранения значений F1-score для каждой глубины дерева
times_depth = [] # Создаем пустой список для хранения времени обучения модели для каждой глубины дерева

# Итерируемся по списку значений глубины дерева
for depth in depth_values:
    start_time = time.time() # Записываем время начала обучения модели
    # Создаем модель случайного леса с указанной глубиной дерева, количеством деревьев и фиксированным random_state.
    model = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
    model.fit(X_train, y_train) # Обучаем модель на обучающей выборке
    preds = model.predict(X_test) # Получаем предсказания модели на тестовой выборке

    f1_scores_depth.append(f1_score(y_test, preds)) # Вычисляем F1-score для предсказаний и добавляем его в список
    times_depth.append(time.time() - start_time) # Вычисляем время обучения модели и добавляем его в список

plt.figure(figsize=(12, 6)) # Создаем новую фигуру для графика
plt.plot(depth_values, f1_scores_depth, marker='o')  #Строим график зависимости F1-score от глубины дерева
plt.title('Зависимость F1-score от глубины деревьев') # Устанавливаем заголовок графика
plt.xlabel('Глубина дерева') # Устанавливаем подпись для оси X
plt.ylabel('F1-score') # Устанавливаем подпись для оси Y
plt.grid() # Включаем отображение сетки на графике
plt.show() # Отображаем график

# Исследование зависимости от количества признаков
feature_values = range(1, X.shape[1] + 1) # Создаем список значений количества признаков для исследования
f1_scores_feat = [] # Создаем пустой список для хранения значений F1-score для каждого количества признаков
times_feat = [] # Создаем пустой список для хранения времени обучения модели для каждого количества признаков

for feat in feature_values: # Итерируемся по списку значений количества признаков
    start_time = time.time() # Записываем время начала обучения модели
    model = RandomForestClassifier(max_features=feat, n_estimators=100, random_state=42) # Создаем модель случайного леса с указанным количеством признаков
    model.fit(X_train, y_train) # Обучаем модель на обучающей выборке
    preds = model.predict(X_test) # Получаем предсказания модели на тестовой выборке

    f1_scores_feat.append(f1_score(y_test, preds)) # Вычисляем F1-score для предсказаний и добавляем его в список
    times_feat.append(time.time() - start_time) # Вычисляем время обучения модели и добавляем его в список

plt.figure(figsize=(12, 6)) # Создаем новую фигуру для графика
plt.plot(feature_values, f1_scores_feat, marker='o', color='orange') #Строим график зависимости F1-score от количества признаков
plt.title('Зависимость F1-score от количества признаков')
plt.xlabel('Количество признаков')
plt.ylabel('F1-score')
plt.grid() # Включаем отображение сетки на графике
plt.show() # Отображаем график

# Исследование зависимости от количества деревьев
tree_values = range(10, 210, 10) # Создаем список значений количества деревьев для исследования
f1_scores_trees = [] # Создаем пустой список для хранения значений F1-score для каждого количества деревьев
times_trees = [] # Создаем пустой список для хранения времени обучения модели для каждого количества деревьев

# Итерируемся по списку значений количества деревьев
for n in tree_values:
    start_time = time.time() # Записываем время начала обучения модели
    model = RandomForestClassifier(n_estimators=n, random_state=42) # Создаем модель случайного леса с указанным количеством деревьев и фиксированным random_state
    model.fit(X_train, y_train) # Обучаем модель на обучающей выборке
    preds = model.predict(X_test) # Получаем предсказания модели на тестовой выборке

    f1_scores_trees.append(f1_score(y_test, preds)) # Вычисляем F1-score для предсказаний и добавляем его в список
    times_trees.append(time.time() - start_time) # Вычисляем время обучения модели и добавляем его в список

plt.figure(figsize=(12, 6)) # Создаем новую фигуру для график
plt.plot(tree_values, f1_scores_trees, marker='o', color='green', label='F1-score')# Строим график зависимости F1-score от количества деревьев.
plt.plot(tree_values, times_trees, marker='o', color='red', label='Время обучения') # Строим график зависимости времени обучения от количества деревьев
plt.title('Зависимость F1-score и времени обучения от количества деревьев')
plt.xlabel('Количество деревьев')
plt.ylabel('Значение')
plt.legend()
plt.grid()
plt.show()# Отображаем график

# Определяем словарь с параметрами для модели XGBoost
params = {
    'learning_rate': 0.2, # Скорость обучения
    'max_depth': 3, # Максимальная глубина дерева
    'n_estimators': 150, # Количество деревьев
    'subsample': 0.9, # Доля выборки для обучения каждого дерева
    'colsample_bytree': 0.9, # Доля признаков для обучения каждого дерева
    'objective': 'binary:logistic'
}

start_time = time.time() # Записываем время начала обучения модели XGBoost
xgb_model = XGBClassifier(**params) # Создаем модель XGBoost с указанными параметрами
xgb_model.fit(X_train, y_train) # Обучаем модель на обучающей выборке
xgb_time = time.time() - start_time # Вычисляем время обучения модели XGBoost

xgb_preds = xgb_model.predict(X_test) # Получаем предсказания модели XGBoost на тестовой выборке
xgb_f1 = f1_score(y_test, xgb_preds) # Вычисляем F1-score для предсказаний модели XGBoost

# Вывод информации
print('\nСравнение моделей:')
print(f"Random Forest (лучший F1): {max(f1_scores_depth):.3f}")
print(f"XGBoost F1: {xgb_f1:.3f}")
print(f"\nВремя обучения XGBoost: {xgb_time:.2f} сек")
print(f"Среднее время обучения RF: {np.mean(times_trees):.2f} сек")

plt.figure(figsize=(12, 6)) # Создаем новую фигуру для графика
plt.barh(X.columns, xgb_model.feature_importances_) # Строим горизонтальную столбчатую диаграмму, показывающую важность признаков в модели XGBoost
plt.title('Важность признаков в XGBoost') # Устанавливаем заголовок графика
plt.show() # Отображаем график
