### Data
import pandas as pd
import numpy as np
import pickle
### Graphing
import plotly.graph_objects as go
### Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
## Navbar
from navbar import Navbar
from dash import no_update
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#metrics import classification_report
from sklearn.metrics import classification_report,roc_curve

pkl_filename="Hotel_Bookings.pkl"
with open(pkl_filename, 'rb') as file:
    df_dummies = pickle.load(file)

X = df_dummies.drop(columns=['is_canceled'])
X = X.drop(columns=['arrival_date_year', 'arrival_date_week_number',
       'arrival_date_day_of_month', 'stays_in_weekend_nights',
       'stays_in_week_nights', 'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'booking_changes','required_car_parking_spaces',
       'total_of_special_requests', 'total_guest','reservation_status_Canceled','reservation_status_Check-Out',
       'reservation_status_No-Show'])
Y = df_dummies['is_canceled'].values

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state=0)
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED,'https://codepen.io/chriddyp/pen/bWLwgP.css'])
server=app.server
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
app.config.suppress_callback_exceptions = True
nav = Navbar()
header = html.H3(
    'Random Forest'
)
header9 = html.H6('Because of Space and Server Constraint Please use the combinations of parameters given :')
para =  html.P('1- bootstrap=True | max_depth = 100 | max_features = 0.5 | min_samples_leaf = 1 | min_samples_split = 4 | n_estimators=300')
para1 = html.P('2- bootstrap=True | max_depth = 100 | max_features = 0.5 | min_samples_leaf = 3 | min_samples_split = 2 | n_estimators=100')
para2 = html.P('3- bootstrap=True | max_depth = 10  | max_features = 0.5 | min_samples_leaf = 1 | min_samples_split = 2 | n_estimators=100')
para3 = html.P('4- bootstrap=True | max_depth = 10  | max_features = 0.1 | min_samples_leaf = 1 | min_samples_split = 2 | n_estimators=100')
para4 = html.P('5- bootstrap=True | max_depth = 50  | max_features = 1   | min_samples_leaf = 5 | min_samples_split = 8 | n_estimators=200')
dropdown = html.Div([
    html.Label('Bootstrap'),
    dcc.Dropdown(
    id = 'bs',
    options = [{'label': 'True', 'value': 'True'},
               {'label': 'False', 'value': 'False'}],value='True'
    ),
    html.Label('Max number of features'),
    dcc.Dropdown(
    id = 'maxf',
    options = [{'label': '0.5', 'value': 0.5},{'label': '1.0', 'value': 1}],value=0.5
    ),
    html.Label('Max Depth: '),
    dcc.Dropdown(
    id = 'maxd',
    options = [{'label': '4', 'value': 4},
                {'label': '10', 'value': 10},
                 {'label':'100','value':100}],value=4
    ),
    html.Label('minimum samples leaf: '),
    dcc.Dropdown(
    id = 'min_samples_leaf',
    options = [{'label': '1', 'value': 1},
                {'label': '3', 'value': 3}],value=1
    ),
    html.Label('No. of estimators: '),
    dcc.Dropdown(
    id = 'n_est',
    options = [{'label': '100', 'value': 100}],value=100
    ),
    html.Label('Minimum samples split: '),
    dcc.Dropdown(
    id = 'min_samples_split',
    options = [{'label': '2', 'value': 2},
                {'label': '4', 'value': 4}],value=2
    ),
])
output = html.Div(id = 'output',
                children = [],
                )
header1 = html.H3('Classification Report')
output7 = html.Div(id='rforest')
header2 = html.H3('TPR v/s FPR')
output1 = html.Div([dcc.Graph(id='rfgraph')])

def App():
    layout = html.Div([
        nav,
        header,
        header9,
        para,
        para1,
        para2,
        para3,
        para4,
        dropdown,
        output,
        header1,
        output7,
        header2,
        output1,
    ])
    return layout

def randomForest(bs, maxd, maxf, min_samples_leaf,n_est,min_samples_split):
    if bs=='True' and maxd==4 and maxf==0.5 and min_samples_leaf==1 and n_est==100 and min_samples_split==4:
        print("1")
        pkl1_filename = "rf.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    elif bs=='True' and maxd==100 and maxf==0.5 and min_samples_leaf==3 and n_est==100 and min_samples_split==2:
        print("2")
        pkl1_filename = "rf1.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    elif bs == 'True' and maxd == 10 and maxf == 0.5 and min_samples_leaf == 1 and n_est == 100 and min_samples_split == 2:
        print("3")
        pkl1_filename = "rf2.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    elif bs == 'True' and maxd == 10 and maxf == 0.1 and min_samples_leaf == 1 and n_est == 100 and min_samples_split == 2:
        print("4")
        pkl1_filename = "rf3.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    else:
        pkl1_filename = "rf4.pkl"
        print("5")
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
    print("Hello World")
    return generate_table(df)

def update_graphrf(bs, maxd, maxf, min_samples_leaf,n_est,min_samples_split):
    if bs=='True' and maxd==4 and maxf==0.5 and min_samples_leaf==1 and n_est==100 and min_samples_split==4:
        pkl1_filename = "rf.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    elif bs == 'True' and maxd == 100 and maxf == 0.5 and min_samples_leaf == 3 and n_est == 100 and min_samples_split == 2:
        pkl1_filename = "rf1.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    elif bs == 'True' and maxd == 10 and maxf == 0.5 and min_samples_leaf == 1 and n_est == 100 and min_samples_split == 2:
        pkl1_filename = "rf2.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    elif bs == 'True' and maxd == 10 and maxf == 0.1 and min_samples_leaf == 1 and n_est == 100 and min_samples_split == 2:
        pkl1_filename = "rf3.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    else:
        pkl1_filename = "rf4.pkl"
        with open(pkl1_filename, 'rb') as file1:
            rf_model = pickle.load(file1)
        y_pred = rf_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    gp={
        'data': [go.Scatter(
            x=fpr,
            y=tpr,
            text= "ROC Curve for Random Forest",
            mode='lines',
            marker={
                'size': 15,
                'opacity': 0.5,
                'colorscale':'Viridis',
                'line': {'width': 0.5, 'color': 'white'},
                'showscale':True
            }
        )],
        'layout': {
            'xaxis':{
                'title': 'FPR',
                'type': 'linear'
            },
            'yaxis':{
                'title': 'TPR',
                'type': 'linear'
            },
            'margin':{'l': 40, 'b': 40, 't': 10, 'r': 0},
            'hovermode':'closest'
        }
    }
    return gp

