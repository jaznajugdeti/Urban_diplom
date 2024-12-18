import pandas as pd
from sklearn.datasets import load_wine
import plotly.express as px
import plotly.graph_objects as go

# Шаг 1: Загружаем данные из библиотеки Wine
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['target'] = wine.target
data['target_name'] = data['target'].apply(lambda x: wine.target_names[x])

# Шаг 2: Строим графики на базе датасета Wine
# График 1. 3D Scatter Plot (Flavanoids vs Color Intensity vs Alcohol)
# задаем, что откладываем по оси х, у, z, название, цвет
fig1 = px.scatter_3d(data,
                     x='flavanoids',
                     y='color_intensity',
                     z='alcohol',
                     color='target_name',
                     title="3D Scatter Plot: Flavanoids vs Color Intensity vs Alcohol",
                     labels={'target_name': 'Wine Type'},
                     symbol='target_name')
# выводим график
fig1.show()

# График 2. Pairplot (scatter_matrix):
# задаем название, цвет, выбираем данные
fig2 = px.scatter_matrix(data,
                         dimensions=['alcohol', 'malic_acid', 'flavanoids', 'color_intensity'],
                         color='target_name',
                         title="Pairplot (Scatter Matrix) of Wine Dataset",
                         labels={'target_name': 'Wine Type'})
fig2.update_traces(diagonal_visible=False)
# выводим график
fig2.show()

# График 3. Boxplot - ящик с усами: (Alcohol Content)
# задаем, что откладываем по оси х, у, название, цвет
fig3 = px.box(data,
              x='target_name',
              y='alcohol',
              color='target_name',
              title="Boxplot of Alcohol Content by Wine Type",
              labels={'target_name': 'Wine Type', 'alcohol': 'Alcohol Content (%)'})
# выводим график
fig3.show()

# График 4. Violin Plot: (Malic Acid)
# задаем, что откладываем по оси х, у, название, цвет
fig4 = px.violin(data,
                 x='target_name',
                 y='malic_acid',
                 color='target_name',
                 box=True, # Добавить boxplot
                 points="all", # Отображение всех точек
                 title="Violin Plot of Malic Acid Content by Wine Type",
                 labels={'target_name': 'Wine Type', 'malic_acid': 'Malic Acid Content'})
# выводим график
fig4.show()

# График 5: Heatmap корреляционной матрицы - цветовую даиграмму:

correlation_matrix = data.iloc[:, :-2].corr()
# задаем, что откладываем по оси х, у, z, название
fig5 = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='Viridis'))
# обновляем названия
fig5.update_layout(
    title="Correlation Matrix Heatmap",
    xaxis_title="Features",
    yaxis_title="Features")
# выводим график
fig5.show()