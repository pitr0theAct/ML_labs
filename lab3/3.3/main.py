import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_curve, roc_curve, roc_auc_score

try:
    df = pd.read_csv("Titanic.csv")
except Exception as e:
    print(f"Ошибка при загрузке датасета: {e}")
    exit()

print("\nПервые 5 строк до предобработки:")
print(df.head())


initial_rows = len(df)

df.dropna(inplace=True)

columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df.drop(columns=columns_to_drop, inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df['Embarked'] = df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

print("\nПервые 5 строк после предобработки:")
print(df.head())

final_rows = len(df)
lost_rows = initial_rows - final_rows
lost_percentage = (lost_rows / initial_rows) * 100
print(f"\nВсего строк до чистки: {initial_rows}")
print(f"Строк после чистки: {final_rows}")
print(f"Процент потерянных данных: {lost_percentage:.2f} %")

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nДанные разделены на обучающую ({len(X_train)} строк) и тестовую ({len(X_test)} строк) выборки")

model = LogisticRegression(random_state=0, max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nОценка модели (Метрики классификации)")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(conf_matrix)

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR")
plt.grid(True)
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_roc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label=f'AUC = {auc_roc:.4f}')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.legend()
plt.grid(True)
plt.show()

print("\nОценка влияния признака 'Embarked':")

X_no_embarked = df.drop(['Survived', 'Embarked'], axis=1)
X_no_embarked_train, X_no_embarked_test, y_train, y_test = train_test_split(X_no_embarked, y, test_size=0.2,
                                                                            random_state=42)

model_no_embarked = LogisticRegression(random_state=0, max_iter=200)
model_no_embarked.fit(X_no_embarked_train, y_train)

y_pred_no_embarked = model_no_embarked.predict(X_no_embarked_test)
accuracy_no_embarked = accuracy_score(y_test, y_pred_no_embarked)

print(f"Точность модели без признака 'Embarked': {accuracy_no_embarked:.3f}")

print("Разница в точности:", accuracy - accuracy_no_embarked)
