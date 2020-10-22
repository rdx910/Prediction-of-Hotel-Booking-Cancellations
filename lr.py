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
from sklearn.linear_model import LogisticRegression
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
    'Logistic Regression'
)
header9 = html.H6('Because of Space and Server Constraint Please use the combinations of parameters given :')
para =  html.P('1- C == 1    | penalty == L2 | Max number of iterations == 10000')
para1 = html.P('2- C == 0.1  | penalty == L2 | Max number of iterations == 10000')
para2 = html.P('3- C == 5    | penalty == L1 | Max number of iterations == 10000')
para3 = html.P('4- C == 5    | penalty == L2 | Max number of iterations == 10000')
para4 = html.P('5- C == 10   | penalty == l2 | Max number of iterations == 10000')
dropdown1 = html.Div([
    html.Label('C: '),
    dcc.Dropdown(
    id='C',
        options=[

                {'label': '0.1', 'value': 0.1},
                {'label': '1.0', 'value': 1},
                {'label': '5.0', 'value': 5}
            ],
        value=0.1
        ),
    html.Label('Penalty:'),
    dcc.Dropdown(
    id='penalty',
            options=[

                {'label': 'none', 'value': 'none'},
                {'label': 'l2', 'value': 'l2'}
            ],
            value='l2'
        ),
html.Label('Max number of iterations'),
    dcc.Dropdown(
    id='solver',
            options=[

                {'label': '10000', 'value': 10000}
            ],
            value=10000
        ),

])
output2 = html.Div(id = 'output',
                children = [],
                )
header1 = html.H3('Classification Report')
output6 = html.Div(id='LReg')
header2 = html.H3('TPR v/s FPR')
output3 = html.Div([dcc.Graph(id='lrgraph')])

def App1():
    layout = html.Div([
        nav,
        header,
        header9,
        para,
        para1,
        para2,
        para3,
        para4,
        dropdown1,
        output2,
        header1,
        output6,
        header2,
        output3,
    ])
    return layout

def logisticRegression(C,penalty,solver):
    if C==1 and penalty=='L2' and solver=='10000':
        pkl2_filename = "lr.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    elif C == 0.1 and penalty == 'L2' and solver == '10000':
        pkl2_filename = "lr1.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    elif C == 5 and penalty == 'none' and solver == '10000':
        pkl2_filename = "lr2.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    elif C == 5 and penalty == 'L2' and solver == '10000':
        pkl2_filename = "lr3.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()

    else:
        pkl2_filename = "lr4.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
    return generate_table(df),roc_auc_score(y_test,y_pred)


def update_graphlr(C,penalty,solver):
    if C == 1 and penalty == 'L2' and solver == '10000':
        pkl2_filename = "lr.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    elif C == 0.1 and penalty == 'L2' and solver == '10000':
        pkl2_filename = "lr1.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)

    elif C == 5 and penalty == 'none' and solver == '10000':
        pkl2_filename = "lr2.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)

    elif C == 5 and penalty == 'L2' and solver == '10000':
        pkl2_filename = "lr3.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)

    else:
        pkl2_filename = "lr4.pkl"
        with open(pkl2_filename, 'rb') as file2:
            lr_model = pickle.load(file2)
        y_pred = lr_model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
    gp = {
        'data': [go.Scatter(
            x=fpr,
            y=tpr,
            text="ROC Curve for Logistic Regression",
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

