import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go

# Шаг 1: Загружаем данные из библиотеки Iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x])

# Шаг 2: Строим графики с помощью Ploty
# График 1. 3D Scatter Plot - рассеянные точки:
# задаем, что откладываем по оси х, у, z, название, цвет

fig1 = px.scatter_3d(data,
                     x='sepal length (cm)',
                     y='sepal width (cm)',
                     z='petal length (cm)',
                     color='species_name',
                     title="3D Scatter Plot of Iris Dataset",
                     labels={"species_name": "Species"})
# выводим график
fig1.show()

# График 2. Pairplot (scatter_matrix):
# задаем, ширину и высоту фигуры, название, цвет
fig2 = px.scatter_matrix(data,
                         dimensions=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                         color='species_name',
                         title="Scatter Matrix of Iris Dataset",
                         labels={"species_name": "Species"})
# выводим график
fig2.show()

# График 3. Boxplot - ящик с усами:
# задаем, что откладываем по оси х, у, название, цвет
fig3 = px.box(data,
              x='species_name',
              y='petal length (cm)',
              color='species_name',
              title="Boxplot of Petal Length by Species",
              labels={"species_name": "Species", "petal length (cm)": "Petal Length (cm)"})
# выводим график
fig3.show()

# График 4. Violin Plot:
# задаем, что откладываем по оси х, у, название, цвет
fig4 = px.violin(data,
                 x='species_name',
                 y='sepal width (cm)',
                 color='species_name',
                 box=True, # Добавлям boxplot
                 points='all', # Добавляем точки
                 title="Violin Plot of Sepal Width by Species",
                 labels={"species_name": "Species", "sepal width (cm)": "Sepal Width (cm)"})
# выводим график
fig4.show()

# График 5. Heatmap корреляционной матрицы - цветовую даиграмму:
correlation_matrix = data.iloc[:, :-2].corr()
# задаем, что откладываем по оси х, у, z, название
fig5 = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='Viridis'
))
fig5.update_layout(title="Correlation Matrix Heatmap")
# выводим график
fig5.show()