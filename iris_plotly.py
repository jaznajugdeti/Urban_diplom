import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go

''' 
Скачиваем необходимые библиотеки.
Далее необходимо провести ряд изменений в датасете для удобства:
iris = load_iris() - скачиванем датасет;
data['species'] = iris.target- добавляем целевую переменную;
data = pd.DataFrame(data=iris.data, columns=iris.feature_names) — это превращение датасета о цветах в датафрейм
с использованием библиотеки pandas.
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x]) - Этот код добавляет столбец с названиями 
видов ирисов на основе целевой переменной (iris.target);
'''

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x])

''' 
Далее мы начинаем строить графики.
График 1. 3D Scatter Plot - рассеянные точки:
px.scatter_3d() – использует упрощённый интерфейс к библиотеке Plotly Express и позволяет создавать диаграммы рассеяния 
с помощью одного вызова функции, автоматически обрабатывая многие аспекты макета графика:
data  — содержит набор данные, x, y, z — имена столбцов DataFrame, соответствующие координатам x, y и z точек данных, 
color  используемый цвет для точек, labels – это аргумент, который позволяет задать метки осей, title - название графика.
После чего выводим график - fig1.show().
'''

fig1 = px.scatter_3d(data,
                     x='sepal length (cm)',
                     y='sepal width (cm)',
                     z='petal length (cm)',
                     color='species_name',
                     title="3D Scatter Plot of Iris Dataset",
                     labels={"species_name": "Species"})
fig1.show()

''' 
Строим следующие графики.
График 2. Pairplot (scatter_matrix):
px.scatter_matrix() - код для построения графика со следующими условиями:
dimensions — список столбцов, по которым будет отображаться матрица, color — цвет для точек данных,
title – название графика и labels – это аргумент, который позволяет задать метки осей.
После чего выводим график - fig2.show().
'''

fig2 = px.scatter_matrix(data,
                         dimensions=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                         color='species_name',
                         title="Scatter Matrix of Iris Dataset",
                         labels={"species_name": "Species"})
fig2.show()

''' 
Строим следующие графики.
График 3. Boxplot - ящик с усами:
px.box() - код для построения графика со следующими условиями:
data  — содержит набор данные, x, y, — имена столбцов DataFrame, соответствующие координатам x, y точек данных, 
color - используемый цвет для точек, labels – это аргумент, который позволяет задать метки осей, title - название графика.
После чего выводим график - fig3.show().
'''


fig3 = px.box(data,
              x='species_name',
              y='petal length (cm)',
              color='species_name',
              title="Boxplot of Petal Length by Species",
              labels={"species_name": "Species", "petal length (cm)": "Petal Length (cm)"})
fig3.show()

''' 
Строим следующие графики.
График 4. Violin Plot:
px.violin() - код для построения графика со следующими условиями:
data — DataFrame, который содержит данные для построения графика, x  и у — имя столбца или подобный массиву объект, 
который представляет переменную для построения на оси x и у, color – задаваемый цвет и labels – параметр, который 
представляет собой словарь для сопоставления имён столбцов с метками, что позволяет переопределить стандартные метки осей, 
box=True – форма создания ящика и points='all' – отображение всех точек на  графике и title – название графика
После чего выводим график - fig4.show().
'''


fig4 = px.violin(data,
                 x='species_name',
                 y='sepal width (cm)',
                 color='species_name',
                 box=True, # Добавлям boxplot
                 points='all', # Добавляем точки
                 title="Violin Plot of Sepal Width by Species",
                 labels={"species_name": "Species", "sepal width (cm)": "Sepal Width (cm)"})
fig4.show()

''' 
Строим следующие графики.
График 5. Heatmap корреляционной матрицы - цветовую даиграмму:
При этом, вначале необходимо построить  двумерную матрицу случайных значений с помощью библиотеки numpy. 
Затем создаётся тепловая карта с помощью функции go.Heatmap() и предоставляются матрица значений и цветовая шкала. 
После этого создаются макет и фигура с помощью функций  go.Figure(). В конце фигура отображается с помощью функции show():
correlation_matrix = data.iloc[:, :-2].corr()
go.Figure(data=go.Heatmap() - код для создания графика тепловой карты со следующими параметрами: data – перечень 
используемых данных, х,y,z – указывают на метки столбцов и строк, соответственно, 
colorscale в этом выражении указывает цветовую шкалу для тепловой карты.
Также с помощью кода можно изменить или добавить названия  графика и осей х и у -
fig5.update_layout( title=" ", xaxis_title="", yaxis_title="")
После чего выводим график - fig5.show().
'''


correlation_matrix = data.iloc[:, :-2].corr()

fig5 = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='Viridis'
))
fig5.update_layout(title="Correlation Matrix Heatmap")
fig5.show()