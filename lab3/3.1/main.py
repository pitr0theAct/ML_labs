import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

print("Названия сортов:", iris.target_names)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter1 = plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'], cmap='viridis')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal Length vs Sepal Width')
handles1, _ = scatter1.legend_elements()
legend1 = plt.legend(handles1, target_names, title="Classes")

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['target'], cmap='viridis')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Petal Length vs Petal Width')
handles2, _ = scatter2.legend_elements()
legend2 = plt.legend(handles2, target_names, title="Classes")

plt.tight_layout()
plt.show()


sns.pairplot(df, hue='target', palette='viridis')  # hue красит точки по таргету
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

df_binary1 = df[df['target'] != 2].copy()

df_binary2 = df[df['target'] != 0].copy()
df_binary2['target'] = df_binary2['target'].replace({1: 0, 2: 1})

print("\nПодготовлены 2 бинарных датасета.")

print("\nКлассификация: setosa vs versicolor")

X1 = df_binary1.drop('target', axis=1)
y1 = df_binary1['target']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25,
                                                        random_state=0)
model1 = LogisticRegression(random_state=0)

model1.fit(X1_train, y1_train)

y1_pred = model1.predict(X1_test)

accuracy1 = accuracy_score(y1_test, y1_pred)
print(f"Точность модели (setosa vs versicolor) {accuracy1:.4f}")

print("\nКлассификация: versicolor vs virginica")

X2 = df_binary2.drop('target', axis=1)
y2 = df_binary2['target']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=0)

model2 = LogisticRegression(random_state=0)

model2.fit(X2_train, y2_train)

y2_pred = model2.predict(X2_test)

accuracy2 = accuracy_score(y2_test, y2_pred)
print(
    f"Точность модели (versicolor vs virginica): {accuracy2 :.4f}")

print("\nКлассификация случайного датасета")

X_synth, y_synth = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                       n_informative=2, random_state=1, n_clusters_per_class=1)

plt.figure(figsize=(6, 6))
plt.scatter(X_synth[:, 0], X_synth[:, 1], c=y_synth, cmap='viridis', marker='o',
            edgecolors='k')
plt.title("Сгенерированный датасет")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.show()

X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(X_synth, y_synth, test_size=0.25,
                                                                            random_state=0)
model_synth = LogisticRegression(random_state=0)

model_synth.fit(X_synth_train, y_synth_train)

y_synth_pred = model_synth.predict(X_synth_test)

accuracy_synth = accuracy_score(y_synth_test, y_synth_pred)
print(f"Точность модели (случайный датасет): {accuracy_synth:.4f}")
