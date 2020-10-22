import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from rf import App, update_graphrf,randomForest
from lr import App1, update_graphlr,logisticRegression
from gbm import App2, update_graphgb,gradientBoosting
from homepage import Homepage
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED,'https://codepen.io/chriddyp/pen/bWLwgP.css'])
server=app.server
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id = 'url', refresh = False),
    html.Div(id = 'page-content')
])
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
    )

def display_page(pathname):
    if pathname == '/model':
        return App()
    if pathname == '/Logistic':
        return App1()
    if pathname == '/gbm':
        return App2()
    else:
        return Homepage()

@app.callback(

        Output('rforest','children'),
        [Input('bs','value'),
        Input('maxd','value'),
        Input('maxf','value'),
        Input('min_samples_leaf','value'),
        Input('n_est','value'),
        Input('min_samples_split','value')]
    )

def update_model2(bs, maxd, maxf, min_samples_leaf,n_est,min_samples_split):
    rf = randomForest(bs, maxd, maxf, min_samples_leaf,n_est,min_samples_split)
    return rf

@app.callback(
     Output('rfgraph', 'figure'),
    [Input('bs', 'value'),
     Input('maxf', 'value'),
     Input('maxd', 'value'),
     Input('min_samples_leaf', 'value'),
     Input('n_est', 'value'),
     Input('min_samples_split', 'value')]
     )
def update_graph(bs, maxd, maxf, min_samples_leaf,n_est,min_samples_split):
    graph = update_graphrf(bs, maxd, maxf, min_samples_leaf,n_est,min_samples_split)
    return graph


@app.callback(
     Output('LReg', 'children'),
    [Input('C', 'value'),
     Input('penalty', 'value'),
     Input('solver', 'value')]
     )
def update_model1(C,penalty,solver):
    lr = logisticRegression(C,penalty,solver)
    return lr

@app.callback(
     Output('lrgraph', 'figure'),
    [Input('C', 'value'),
     Input('penalty', 'value'),
     Input('solver', 'value')]
     )

def update_graph1(C,penalty,solver):
    graph1 = update_graphlr(C,penalty,solver)
    return graph1

@app.callback(
   Output('GBModel','children'),

    [Input('learning_rate','value'),
    Input('gn_est','value'),
     Input('gmaxd','value')
    ]
)

def update_model3(learning_rate,gn_est,gmaxd):
    gb = gradientBoosting(learning_rate,gn_est,gmaxd)
    return gb

@app.callback(
    Output('gbgraph', 'figure'),
    [Input('learning_rate','value'),
    Input('gn_est','value'),
     Input('gmaxd','value')
     ]
     )
def update_graph2(learning_rate,gn_est,gmaxd):
    graph2 = update_graphgb(learning_rate,gn_est,gmaxd)
    return graph2

if __name__ == "__main__":
    app.run_server()