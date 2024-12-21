import pandas as pd
from sklearn.datasets import load_wine
import plotly.express as px
import plotly.graph_objects as go

''' 
Скачиваем необходимые библиотеки.
Далее необходимо провести ряд изменений в датасете для удобства:
wine = load_wine() - скачиванем датасет;
data['target'] = wine.target - добавляем целевую переменную;
data = pd.DataFrame(wine.data, columns=wine.feature_names) — это превращение датасета о вине в датафрейм
с использованием библиотеки pandas.
data['target_name'] = data['target'].apply(lambda x: wine.target_names[x]) - Это код для добавления нового столбца 
с именами целей для лучшей читаемости. Он использует метод apply() для применения лямбда-функции к столбцу «target» и 
получения значений из объекта wine.target_names по этому столбцу;

'''

wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['target'] = wine.target
data['target_name'] = data['target'].apply(lambda x: wine.target_names[x])

''' 
Далее мы начинаем строить графики.
График 1. 3D Scatter Plot - рассеянные точки - (Flavanoids vs Color Intensity vs Alcohol):
px.scatter_3d() – использует упрощённый интерфейс к библиотеке Plotly Express и позволяет создавать диаграммы рассеяния 
с помощью одного вызова функции, автоматически обрабатывая многие аспекты макета графика:
data  — содержит набор данные, x, y, z — имена столбцов DataFrame, соответствующие координатам x, y и z точек данных, 
color  используемый цвет для точек, labels – это аргумент, который позволяет задать метки осей, title - название графика,
title – название графика и symbol – это параметр, который указывает символ маркера.
После чего выводим график - fig1.show().
'''

fig1 = px.scatter_3d(data,
                     x='flavanoids',
                     y='color_intensity',
                     z='alcohol',
                     color='target_name',
                     title="3D Scatter Plot: Flavanoids vs Color Intensity vs Alcohol",
                     labels={'target_name': 'Wine Type'},
                     symbol='target_name')
fig1.show()

''' 
Строим следующие графики.
График 2. Pairplot (scatter_matrix):
px.scatter_matrix() - код для построения графика со следующими условиями:
dimensions — список столбцов, по которым будет отображаться матрица, color — цвет для точек данных,
title – название графика и labels – это аргумент, который позволяет задать метки осей.
fig2.update_traces(diagonal_visible=False) - позволяет сделать так, чтобы на графике не была видна диагональ.
После чего выводим график - fig2.show().
'''

fig2 = px.scatter_matrix(data,
                         dimensions=['alcohol', 'malic_acid', 'flavanoids', 'color_intensity'],
                         color='target_name',
                         title="Pairplot (Scatter Matrix) of Wine Dataset",
                         labels={'target_name': 'Wine Type'})
fig2.update_traces(diagonal_visible=False)
fig2.show()

''' 
Строим следующие графики.
График 3. Boxplot - ящик с усами - (Alcohol Content):
px.box() - код для построения графика со следующими условиями:
data  — содержит набор данные, x, y, — имена столбцов DataFrame, соответствующие координатам x, y точек данных, 
color - используемый цвет для точек, labels – это аргумент, который позволяет задать метки осей, title - название графика.
После чего выводим график - fig3.show().
'''

fig3 = px.box(data,
              x='target_name',
              y='alcohol',
              color='target_name',
              title="Boxplot of Alcohol Content by Wine Type",
              labels={'target_name': 'Wine Type', 'alcohol': 'Alcohol Content (%)'})
fig3.show()

''' 
Строим следующие графики.
График 4. Violin Plot - (Malic Acid):
px.violin() - код для построения графика со следующими условиями:
data — DataFrame, который содержит данные для построения графика, x  и у — имя столбца или подобный массиву объект, 
который представляет переменную для построения на оси x и у, color – задаваемый цвет и labels – параметр, который 
представляет собой словарь для сопоставления имён столбцов с метками, что позволяет переопределить стандартные метки осей, 
box=True – форма создания ящика и points='all' – отображение всех точек на  графике и title – название графика
После чего выводим график - fig4.show().
'''

fig4 = px.violin(data,
                 x='target_name',
                 y='malic_acid',
                 color='target_name',
                 box=True, # Добавить boxplot
                 points="all", # Отображение всех точек
                 title="Violin Plot of Malic Acid Content by Wine Type",
                 labels={'target_name': 'Wine Type', 'malic_acid': 'Malic Acid Content'})

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
    colorscale='Viridis'))
fig5.update_layout(
    title="Correlation Matrix Heatmap",
    xaxis_title="Features",
    yaxis_title="Features")
fig5.show()