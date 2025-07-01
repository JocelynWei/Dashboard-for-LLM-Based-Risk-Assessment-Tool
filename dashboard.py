import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
import os
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np

# === Data Loading
import os
import pandas as pd

csv_path = r"C:\Users\DELL\xxxx\data.csv" ## change the path before using


df = pd.read_csv(csv_path)

valid_risk_types = [
    "Macroeconomic & Geopolitical Risks",
    "Business/Product Segmentation Risks",
    "Industry & Market Risks",
    "Regulatory & Compliance Risks",
    "Financial Risks",
    "Operational & Supply Chain Risks",
    "Technology Risks"
    ]
df = df[df["risk_type"].isin(valid_risk_types)]

all_tickers = sorted(df['ticker'].unique())

print(df)


# === Initialize Dash as app
app = dash.Dash(__name__)

# === Layout Setting
app.layout = html.Div([
    html.H2("LLM-Powered Risk Assessment Dashboard", style={'textAlign': 'center'}),
    
    # Menu
    html.Div([
        html.Label("Select Tickers:", style={'font-weight': 'bold', 'margin-bottom': '10px'}),
        
        # Select All & Clear All Botton
        html.Div([
            html.Button("Select All", id="select-all-btn", n_clicks=0,
                       style={'margin-right': '10px', 'background-color': '#28a745', 
                              'color': 'white', 'border': 'none', 'padding': '5px 10px', 
                              'cursor': 'pointer', 'border-radius': '3px'}),
            html.Button("Clear All", id="clear-all-btn", n_clicks=0,
                       style={'margin-right': '10px', 'background-color': '#dc3545', 
                              'color': 'white', 'border': 'none', 'padding': '5px 10px', 
                              'cursor': 'pointer', 'border-radius': '3px'}),
        ], style={'margin-bottom': '10px'}),
        
        # Display Number of Selected Companies
        html.Div(id='selected-count', style={'margin-bottom': '10px', 'font-size': '14px', 'color': '#666'}),
        
        # Allow Multi-Selection
        dcc.Checklist(
            id='ticker-checklist',
            options=[{'label': ticker, 'value': ticker} for ticker in all_tickers],
            value=[all_tickers[0]] if all_tickers else [],  
            style={'max-height': '200px', 'overflow-y': 'auto', 'border': '1px solid #ddd', 
                   'padding': '10px', 'background-color': '#f8f9fa', 'border-radius': '5px'},
            labelStyle={'display': 'block', 'margin-bottom': '5px', 'cursor': 'pointer'}
        )
    ], style={'margin': '20px', 'border': '1px solid #e0e0e0', 'padding': '15px', 'border-radius': '5px'}),
    
    dcc.Graph(id='network-graph', style={'height': '600px'}),
    html.Hr(),

    html.Div([
        html.Div([
            html.H4("Total Risk Score by Type (Stacked Bar Chart)"),
            dcc.Graph(id='stacked-bar')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H4("Risk Distribution by Type (Radar Chart)"),
            dcc.Graph(id='radar-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ]),
    html.Hr(),
    html.H4("Company Risk Similarity (Cosine Heatmap)", style={'textAlign': 'center'}),
    dcc.Graph(id='cosine-heatmap')
])

# === callback

# 1. Callback Controlled by Botton
@app.callback(
    Output('ticker-checklist', 'value'),
    [Input('select-all-btn', 'n_clicks'),
     Input('clear-all-btn', 'n_clicks')],
    [State('ticker-checklist', 'value')]
)

def update_checklist_selection(select_all_clicks, clear_all_clicks, current_values):
    
    ctx = callback_context
    
    if not ctx.triggered:
        return [all_tickers[0]] if all_tickers else []
    
    # Botton ID 
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    print(f"Botton Clicked: {button_id}")  
    
    if button_id == 'select-all-btn':
        print("Exec Select All")
        return all_tickers
    elif button_id == 'clear-all-btn':
        print("Exec Clear All")
        return []
    
    # Return Current Status by Default
    return current_values

# 2. Update Number of Selected Companies
@app.callback(
    Output('selected-count', 'children'),
    Input('ticker-checklist', 'value')
)
def update_selected_count(selected_values):
    if not selected_values:
        return "No tickers selected"
    return f"Selected: {len(selected_values)} ticker(s)"

# 3. Update Visuals
@app.callback(
    [Output('network-graph', 'figure'),
     Output('stacked-bar', 'figure'),
     Output('radar-chart', 'figure'),
     Output('cosine-heatmap', 'figure')],  
    Input('ticker-checklist', 'value')
)
def update_all_graphs(selected_tickers):
    print(f"Update Dashboard by Selected tickers: {selected_tickers}") 
    
    if not selected_tickers:
        empty_fig = go.Figure().add_annotation(
            text="Please select at least 1 ticker",
            xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor='center', yanchor='middle', showarrow=False,
            font=dict(size=16)
        )
        return empty_fig, empty_fig, empty_fig, empty_fig

    filtered = df[df['ticker'].isin(selected_tickers)]
    color_map = assign_colors(selected_tickers)
    
    return (
        generate_network_figure(filtered),
        generate_stacked_bar(filtered, color_map),
        generate_radar_chart(filtered, color_map),
        generate_cosine_heatmap(df)  
    )

def assign_colors(tickers):
    base_colors = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
    return {t: base_colors[i % len(base_colors)] for i, t in enumerate(tickers)}

###

def generate_network_figure(df):
    if df.empty:
        return go.Figure().add_annotation(
            text="No Data Available",
            xref="paper", yref="paper", x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=20)
        )

    G = nx.Graph()
    for _, row in df.iterrows():
        t = str(row['ticker'])
        tag = str(row['risk_tags'])
        G.add_node(t, type='ticker')
        G.add_node(tag, type='tag')
        G.add_edge(t, tag)

    tag_counts = df.groupby('risk_tags')['ticker'].nunique()
    pos = nx.spring_layout(G, k=0.6, iterations=50)

    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', textposition="bottom center",
        marker=dict(size=[], color=[], line=dict(width=2)),
        hovertext=[], hoverinfo='text'
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
        if G.nodes[node].get("type") == "ticker":
            node_trace['marker']['color'] += ("skyblue",)
            node_trace['marker']['size'] += (30,)
            node_trace['hovertext'] += (f"Ticker: {node}",)
        else:
            count = tag_counts.get(node, 1)
            node_trace['marker']['color'] += ("orange",)
            node_trace['marker']['size'] += (15 + count * 5,)
            node_trace['hovertext'] += (f"Tag: {node}<br>Shared by {count} tickers",)

    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        tag = edge[0] if G.nodes[edge[0]]["type"] == "tag" else edge[1]
        count = tag_counts.get(tag, 1)
        color = {1: "#888", 2: "#ff7f0e", 3: "#d62728"}.get(count, "#9467bd")
        edge_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode='lines',
            line=dict(width=max(1, count), color=color),
            hoverinfo='none', showlegend=False
        ))

    return go.Figure(data=edge_trace + [node_trace], layout=go.Layout(
        title="Ticker-Tag Network",
        showlegend=False,
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        plot_bgcolor='white'
    ))

def generate_stacked_bar(df, color_map):
    df_unique = df[['ticker', 'risk_type', 'risk_score']].drop_duplicates()
    agg_df = df_unique.groupby(['risk_type', 'ticker'])['risk_score'].sum().reset_index()

    bars = []
    for ticker in agg_df['ticker'].unique():
        df_t = agg_df[agg_df['ticker'] == ticker]
        bars.append(go.Bar(
            x=df_t['risk_type'], y=df_t['risk_score'],
            name=ticker, marker_color=color_map.get(ticker, 'gray')
        ))

    return go.Figure(data=bars, layout=go.Layout(
        barmode='stack',
        xaxis_title="Risk Type",
        yaxis_title="Total Risk Score",
        plot_bgcolor='white',
        hovermode='x unified'
    ))

def generate_radar_chart(df, color_map):
    
    df_clean = df[['ticker', 'risk_type', 'risk_score']].drop_duplicates()

    all_risk_types = sorted(df_clean['risk_type'].unique())
    fig = go.Figure()

    for ticker in sorted(df_clean['ticker'].unique()):
        ticker_data = df_clean[df_clean['ticker'] == ticker]
        
        scores = []
        for risk_type in all_risk_types:
            match = ticker_data[ticker_data['risk_type'] == risk_type]
            scores.append(match['risk_score'].iloc[0] if not match.empty else 0)
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=all_risk_types,
            fill='toself',
            name=ticker,
            line=dict(color=color_map.get(ticker, 'gray')),
            connectgaps=False
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, df_clean['risk_score'].max() * 1.1]),
            angularaxis=dict(rotation=90, direction='clockwise')
        ),
        showlegend=True
    )
    return fig


def generate_cosine_heatmap(df):
    binary_matrix = pd.crosstab(df['ticker'], df['risk_tags']).astype(int)
    similarity_matrix = cosine_similarity(binary_matrix)
    
    mean_similarities = []
    for i in range(len(similarity_matrix)):
        others_sim = np.concatenate([similarity_matrix[i][:i], similarity_matrix[i][i+1:]])
        mean_sim = np.mean(others_sim)
        mean_similarities.append(mean_sim)
    
    sorted_indices = np.argsort(mean_similarities)[::-1]
    tickers = binary_matrix.index[sorted_indices]
    sorted_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
    
    fig = go.Figure(data=go.Heatmap(
        z=sorted_matrix,
        x=tickers,
        y=tickers,
        colorscale='Viridis',
        colorbar=dict(title="Cosine Similarity"),
        hoverongaps=False
    ))
    
    fig.update_layout(
        xaxis_title="Company Ticker",
        yaxis_title="Company Ticker",
        title="Company Risk Similarity",
        xaxis=dict(tickangle=-45),
        width=800,
        height=800
    )
    return fig

# === Activate app
if __name__ == '__main__':
    app.run(debug=True, port=8051)

