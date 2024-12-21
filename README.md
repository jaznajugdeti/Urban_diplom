#_**Название проекта:**_
**Сравнение различных библиотек для визуализации данных: Matplotlib, Seaborn и Plotly:** 
Создать набор визуализаций с использованием Matplotlib, Seaborn и Plotly, сравнить их функциональность и удобство использования.

#_**Краткое описание библиотек:**_
Библиотека Matplotlib — пакет для визуализации данных в Python, который позволяет работать с данными. 
Matplotlib используется для построения двухмерных графиков и визуализации данных. Это самый популярный и часто используемый пакет инструментов для построения графиков в сообществе Python.
Сама Matplotlib является основой для других библиотек — например, Seaborn позволяет проще создавать графики и имеет больше возможностей для косметического улучшения их внешнего вида.
Seaborn – это библиотека визуализации данных на Python, которая упрощает процесс создания сложных визуализаций. Seaborn тесно интегрирован со структурами данных Pandas, 
позволяя беспрепятственно манипулировать данными и визуализировать их. Библиотека поддерживает визуализацию статистических данных (например, коробчатые диаграммы, тепловые карты). 
Применяется в научных исследованиях и анализе данных, особенно в областях, связанных со статистикой.
Plotly — это библиотека визуализации данных, предоставляющая широкий спектр возможностей интерактивного построения графиков на языке Python. 
Plotly позволяет создавать интерактивные визуализации, включая линейные графики, точечные диаграммы, гистограммы, тепловые карты, трехмерные графики и многое другое. 
Библиотека поддерживает анимацию, возможность создания 3D визуализаций, интеграцию с веб-приложениями. 
Библиотека широко используется в различных областях: анализ данных, научные исследования, финансы, веб-разработка и других.




#_Используемые библиотеки:_
import pandas as pd (верися 2.2.3)
import matplotlib.pyplot as plt (верися 3.10.0)
import plotly.express as px (верися 6.0.0rc0)
import plotly.graph_objects as go
import seaborn as sns (верися 0.13.2)
_ Используемые базы данных:_
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris

#_**Пример использования кода в библиотеке Matplotlib:**_
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

Шаг 1: Загружаем данные из библиотеки Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
Добавляем название видов для удобства
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x])
Шаг 2: Строим графики на базе датасета Iris
График 1. Гистограммы лепестков:
plt.figure(figsize=(10, 6)) # создаем размер графика
выбираем данные,  цвет и прозрачность, и количество столбов
plt.hist(data['sepal length (cm)'], bins=15, alpha=0.7, label='Sepal Length', color='blue')
plt.hist(data['petal length (cm)'], bins=15, alpha=0.7, label='Petal Length', color='orange')
задаем название графика
plt.title("Distribution of Sepal and Petal Lengths", fontsize=16)
задем названия осей,
plt.xlabel("Length (cm)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
создаем сетку, автоматическая настройка промежутка
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#_**Пример использования кода в библиотеке Plotly:**_
Шаг 1: Скачиваем данные из дата сета Wine:
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['target'] = wine.target
data['target_name'] = data['target'].apply(lambda x: wine.target_names[x])
Шаг 2: Строим графики на базе датасета Wine
График 1: 3D Scatter Plot (Flavanoids vs Color Intensity vs Alcohol), задаем параметры, название графика, цвет, символы
fig1 = px.scatter_3d(data,
                     x='flavanoids',
                     y='color_intensity',
                     z='alcohol',
                     color='target_name',
                     title="3D Scatter Plot: Flavanoids vs Color Intensity vs Alcohol",
                     labels={'target_name': 'Wine Type'},
                     symbol='target_name')
fig1.show()

#_**Пример использования кода в библиотеке Seaborn:**_
Шаг 1: Загрузка данных
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x])
Шаг 2: Графики с Seaborn
График 1: Pairplot для всех переменных, задаем дата сет, цветовую гамму, высоту графиков, формы графиков, и сопоставляем аспекты графика разным цветам:
sns.pairplot(data, hue='species_name', palette='Set2', diag_kind='kde', height=2)
plt.suptitle("Pairplot of Iris Dataset", y=1.02, fontsize=16)
plt.show()

#_**Использованные источники:**_
Библиотека Matplotlib в Python: что это такое, примеры построения графиков функций и диаграмм / Skillbox Media
Data Visualization with Seaborn - Python - GeeksforGeeks
User guide and tutorial — seaborn 0.13.2 documentation 
Шпаргалка по визуализации данных в Python с помощью Plotly / Хабр
Как создать интерактивную визуализацию данных с помощью Plotly в R/Python? 


#**_Исполнитель проекта:_**
Милутинович Ксения


