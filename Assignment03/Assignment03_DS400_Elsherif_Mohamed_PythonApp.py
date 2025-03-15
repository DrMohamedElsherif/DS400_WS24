#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

red_wine = pd.read_csv("./data/winequality-red.csv", sep=";")

# Initialize app
app = dash.Dash(__name__)

# UI Layout
app.layout = html.Div([
    # Title
    html.H1("Red Wine Quality Prediction", style={'text-align': 'center'}),

    # Main container 
    html.Div([
        # Sidebar with inputs
        html.Div([
            html.Label("Select Variables for Prediction:"),
            dcc.Checklist(
                id="variables",
                options=[{"label": col, "value": col} for col in red_wine.columns[:-1]],
                value=red_wine.columns[:-1].tolist(),
                style={'height': '300px', 'overflowY': 'scroll'}
            ),
            html.Label("Select Model:"),
            dcc.Dropdown(
                id="model",
                options=[
                    {"label": "KNN", "value": "KNN"},
                    {"label": "Random Forest", "value": "Random Forest"}
                ],
                value="KNN"
            ),
            html.Label("Set Hyperparameter (e.g., k or ntrees):"),
            dcc.Input(id="hyperparam", type="number", value=5, min=1),
            html.Label("Train/Test Split Ratio:"),
            dcc.Slider(
                id="split_ratio",
                min=0.1,
                max=0.9,
                step=0.05,
                value=0.7,
                marks={i / 10: f"{i / 10}" for i in range(1, 10)}
            ),
            html.Label("Set Seed for Reproducibility:"),
            dcc.Input(id="seed", type="number", value=42),
            html.Button("Run Prediction", id="predict", n_clicks=0)
        ], style={'width': '30%', 'padding': '20px', 'box-sizing': 'border-box'}),  # Sidebar style

        # Results section 
        html.Div([
            html.Div(id="correlation-content", style={'margin-bottom': '20px'}),
            html.Div(id="predictions-content", style={'margin-bottom': '20px'}),
            html.Div(id="accuracy-content")
        ], style={'width': '65%', 'padding': '20px', 'box-sizing': 'border-box'})
    ], style={'display': 'flex', 'flex-direction': 'row'})  # Main container with Flexbox layout
])

# Callbacks
@app.callback(
    [Output("correlation-content", "children"),
     Output("predictions-content", "children"),
     Output("accuracy-content", "children")],
    Input("predict", "n_clicks"),
    State("variables", "value"),
    State("model", "value"),
    State("hyperparam", "value"),
    State("split_ratio", "value"),
    State("seed", "value")
)
def run_prediction(n_clicks, variables, model, hyperparam, split_ratio, seed):
    if n_clicks == 0:
        return html.Div("Adjust parameters and click 'Run Prediction' to view results."), \
               html.Div(), \
               html.Div()

    # Data preparation
    X = red_wine[variables]
    y = red_wine["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model == "KNN":
        clf = KNeighborsClassifier(n_neighbors=int(hyperparam))
    else:  # Random Forest
        clf = RandomForestClassifier(n_estimators=int(hyperparam), random_state=seed)

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Correlation plot 
    corr_matrix = X.corr().round(1)
    correlation_fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.columns),
        colorscale="Viridis",
        showscale=True,
        colorbar_title="Correlation"
    )
    correlation_fig.update_traces(hoverinfo='skip', showscale=True)  # Hide the text annotations

    # Predictions plot
    predictions_fig = px.bar(
        x=np.unique(y_pred, return_counts=True)[0],
        y=np.unique(y_pred, return_counts=True)[1],
        labels={"x": "Quality", "y": "Count"},
        title="Predicted Wine Quality Distribution"
    )

    # Results
    return (
        dcc.Graph(figure=correlation_fig),  # Correlation matrix
        dcc.Graph(figure=predictions_fig),  # Predictions bar plot
        html.Div(f"Model Accuracy: {accuracy:.2f}", style={'font-size': '20px', 'color': 'green'})  # Accuracy
    )

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




