import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pickle
import base64



from navbar import Navbar
nav = Navbar()

image_filename = 'assets/download.png'
image_filename1 = 'assets/cust.png' 
image_filename2 = 'assets/map1.png' 
image_filename3 = 'assets/rfeature.png'
image_filename4 = 'assets/leadtime.png'

 
body = dbc.Container(
    [
       dbc.Row(
           [  
               dbc.Col(
                  [
                     html.H2("IST 707 Hotel Demand Analysis and Predicting Cancellations by Group 9"),
                     html.H6("Juilee Salunkhe"),
                     html.H6("Akshay Bhala"),
                     html.H6("Raj Desai"),
                    
                   ],
                  
               ),
               dbc.Col(
                  [
                     html.H4("Exploratory Data Analysis"),
                     html.H5("Leadtime v/s Customer type by cancellations"),
    
                      html.Img(src=image_filename4),
                   ],
                  
               ),
              dbc.Col(
                 [   
                     html.H2(" "),
                     html.H5("Customer Type V/s Cancellations"),
                     html.Img(src=image_filename1),
                        ]
                     ),
              dbc.Col(
                 [
                     html.H2(" "),
                     html.H5("Home Country of Guests"),
                     html.Img(src=image_filename2),
                 ]
                     ),
               dbc.Col(
                 [   
                     html.H2(" "),
                     html.H5("Comparison between Supervised Learning Classification Models"),
                     html.Img(src=image_filename),
                        ]
                     ),
               dbc.Col(
                 [   html.H2(" "),
                     html.H5("Feature Importance"),
                     html.Img(src=image_filename3),
                        ]
                     ),   
                ]
            ),

       ],
className="mt-4",
)
def Homepage():
    layout = html.Div([
    nav,
    body,
    ])
    return layout

#app = dash.Dash(__name__, external_stylesheets = [dbc.themes.UNITED])
#app.layout = Homepage()
#if __name__ == "__main__":
#   app.run_server()
