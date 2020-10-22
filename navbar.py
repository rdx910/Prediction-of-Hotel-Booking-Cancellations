import dash_bootstrap_components as dbc

def Navbar():
        navbar = dbc.NavbarSimple(
           children=[
              dbc.DropdownMenu(
                 nav=True,
                 in_navbar=True,
                 label="Models",
                 children=[
                    dbc.DropdownMenuItem("RF",href="/model"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("LR",href="/Logistic"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("GBM",href="/gbm"),
                          ],
                      ),
                    ],
          brand="Home",
          brand_href="/home",
          sticky="top",
        )
        return navbar