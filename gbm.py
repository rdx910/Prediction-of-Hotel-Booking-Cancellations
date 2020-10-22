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
from sklearn.ensemble import GradientBoostingClassifier as gbm
from sklearn.model_selection import train_test_split
#metrics import classification_report
from sklearn.metrics import classification_report,roc_curve,roc_auc_score
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED,'https://codepen.io/chriddyp/pen/bWLwgP.css'])
server=app.server
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
app.config.suppress_callback_exceptions = True
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

nav = Navbar()
header = html.H3(
    'Gradient Boosting Classifier'
)
header9 = html.H6('Because of Space and Server Constraint Please use the combinations of parameters given :')
para =  html.P('1- n_estimators=300 | max_depth=9 | learning_rate=0.1')
para1 = html.P('2- n_estimators=100 | max_depth=6 | learning_rate=0.1')
para2 = html.P('3- n_estimators=200 | max_depth=3 | learning_rate=0.5')
para3 = html.P('4- n_estimators=300 | max_depth=6 | learning_rate=1.0')
para4 = html.P('5- n_estimators=100 | max_depth=9 | learning_rate=1.0')

dropdown2 = html.Div([
html.Label('learning rate: '),dcc.Dropdown(
            id='learning_rate',
options=[

                {'label': '0.1', 'value': 0.1},
                {'label': '0.5', 'value': 0.5},
                {'label': '1', 'value': 1}
            ],
            value=0.1

        ),

    html.Label('No. of estimators: '),dcc.Dropdown(
            id='gn_est',
    options=[

                {'label': '100', 'value': 100},
                {'label': '200', 'value': 200},
                {'label': '300', 'value': 300}
            ],
            value=100

        ),


html.Label('Max Depth: '),dcc.Dropdown(
            id='gmaxd',
options=[

                {'label': '3', 'value': 3},
                {'label': '6', 'value': 6},
                {'label': '9', 'value': 9}
            ],
            value=3

        ),
])
output4 = html.Div(id = 'output',
                children = [],
                )
header1 = html.H3('Classification Report')
output8 = html.Div(id='GBModel')
header2 = html.H3('TPR v/s FPR')
output5 = html.Div([dcc.Graph(id='gbgraph')])

def App2():
    layout = html.Div([
        nav,
        header,
        header9,
        para,
        para1,
        para2,
        para3,
        para4,
        dropdown2,
        output4,
        header1,
        output8,
        header2,
        output5,
    ])
    return layout

def gradientBoosting(learning_rate,gn_est,gmaxd):
    if learning_rate==0.1 and gn_est==300 and gmaxd==9:
        print("1")
        pkl3_filename = "gB.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    elif learning_rate == 0.1 and gn_est == 100 and gmaxd == 6:
        print("2")
        pkl3_filename = "gB1.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        fpr, tpr, _ = roc_curve(y_test, y_pred)

    elif learning_rate == 0.5 and gn_est == 200 and gmaxd == 3:
        print("3")
        pkl3_filename = "gB2.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    elif learning_rate == 1 and gn_est == 300 and gmaxd == 6:
        print("4")
        pkl3_filename = "gB3.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    else:
        print("5")
        pkl3_filename = "gB4.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        print("Hello World")
    return generate_table(df),"AUC:",roc_auc_score(y_test,y_pred)

def update_graphgb(learning_rate,gn_est,gmaxd):
    if learning_rate==0.1 and gn_est==300 and gmaxd==9:
        pkl3_filename = "gB.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    elif learning_rate==0.1 and gn_est==100 and gmaxd==6:
        pkl3_filename = "gB1.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)

    elif learning_rate == 0.5 and gn_est == 200 and gmaxd == 3:
        pkl3_filename = "gB2.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)

    elif learning_rate == 1 and gn_est == 300 and gmaxd == 6:
        pkl3_filename = "gB3.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)

    else:
        pkl3_filename = "gB4.pkl"
        with open(pkl3_filename, 'rb') as file3:
            gbm_model = pickle.load(file3)
        y_pred = gbm_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    gp = {
        'data': [go.Scatter(
            x=fpr,
            y=tpr,
            text="ROC Curve for GBM",
            mode='lines',
            marker={
                'size': 15,
                'opacity': 0.5,
                'colorscale': 'Viridis',
                'line': {'width': 0.5, 'color': 'white'},
                'showscale': True
            }
        )],
        'layout': {
            'xaxis': {
                'title': 'FPR',
                'type': 'linear'
            },
            'yaxis': {
                'title': 'TPR',
                'type': 'linear'
            },
            'margin': {'l': 40, 'b': 40, 't': 10, 'r': 0},
            'hovermode': 'closest'
        }
    }
    return gp

