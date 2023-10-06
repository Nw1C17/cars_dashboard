import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import plotly.express as px


import dash
from dash import dcc
from dash import html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

sales = pd.read_csv('Car_sales.csv')

sales.rename(columns={'__year_resale_value':'Year_resale_value'}, inplace=True)

plt.figure(figsize=(12,6))
sns.heatmap(sales.corr(numeric_only=True), cmap="YlGnBu", annot=True)


def Prices():
    prices = sales['Price_in_thousands']
    counts, bins = np.histogram(prices, bins=range(0, 80, 5))
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig = px.bar(x=bins, y=counts, labels={'x': 'prices', 'y': 'count'})
    return fig

def Horsepowers():
    data = sales['Horsepower']
    counts, bins = np.histogram(data, bins=range(0, 450, 50))
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig2 = px.bar(x=bins, y=counts, labels={'x': 'horsepower', 'y': 'count'})
    return fig2

def Manufacture_Sales():
    data = sales.groupby('Manufacturer')[
        'Sales_in_thousands'].sum().reset_index()
    data_sorted = data.sort_values(by='Sales_in_thousands', ascending=True)
    fig = px.bar(data_sorted, x='Sales_in_thousands', y='Manufacturer',
                 text_auto='.2s',
                 title="Продажи по производителям")
    return fig

def Model_Sales(Manufacturer):
    data = sales.loc[sales["Manufacturer"]==Manufacturer].groupby('Model')[
        ['Sales_in_thousands']].sum().reset_index()
    data_sorted = data.sort_values(by='Sales_in_thousands', ascending=False)
    fig2 = px.bar(data_sorted, x='Sales_in_thousands', y='Model',
                  text_auto='.2s',
                  title="Продажи по моделям")

    return fig2

app.layout = html.Div(children=[
       html.Div([
           dcc.Graph(id='fig1'),
       ]) ,
        html.Div([
           dcc.Graph(id='fig2'),
       ]) ,
       html.Div([
           html.H6('Модели автомобилей'),
           dcc.Dropdown(
               id='Manufacturer',
               value="Ford",
               options={i: i for i in sales["Manufacturer"].unique()}
           ),
        html.Div([
           dcc.Graph(id='fig3'),
        ]) ,
       ])
])

@app.callback(
   dash.dependencies.Output('fig1', 'figure'),
    dash.dependencies.Output('fig2', 'figure'),
    dash.dependencies.Output('fig3', 'figure'),
   [dash.dependencies.Input('Manufacturer', 'value')])
def output_fig(Manufacturer):
    fig1 = Manufacture_Sales()
    fig2 = Model_Sales(Manufacturer)
    fig3 = px.imshow(sales[['Engine_size', 'Horsepower',
                            'Price_in_thousands']].corr(), title="Корреляция между лошадинымм силами и размером двигателя")
    return fig1, fig2, fig3

if __name__ == '__main__':
   app.run_server(debug=True)


