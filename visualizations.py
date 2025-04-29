from bokeh.plotting import figure
from bokeh.embed import components
import numpy as np
import pronouncing
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, CategoricalColorMapper, DatetimeTickFormatter
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Category10, Category20, Viridis256
from bokeh.layouts import column
#from sklearn.cluster import KMeans
import networkx as nx
import plotly
import plotly.graph_objects as go
import random
import json
import pandas as pd
from bokeh.models import ColorBar
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import linguistics
import social
import sentimentemotion
import trends
import search

def related_network_graph(urban_dict, n=50):
  # Initialize the graph
  G = nx.Graph()

  # Select n random slang words
  random_slang = random.sample(list(urban_dict.keys()), n)

  # Draw edges
  for slang in random_slang:
    for related in urban_dict[slang]['top_5_entries'][0]['related']:
      G.add_edge(slang, related)

  # Prepare the data for Plotly
  pos = nx.spring_layout(G, k=0.5)  # Layout for node positions
  edges = G.edges()
  edge_x = []
  edge_y = []
  for edge in edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

  # Create edge trace
  edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='gray'),
    hoverinfo='none',
    mode='lines'
  )

  # Create node trace
  node_x = []
  node_y = []
  for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

  # Color bar scale
  degrees = [G.degree(node) for node in G.nodes()]
  min_degree = min(degrees)
  max_degree = max(degrees)
  tickvals = list(range(min_degree, max_degree + 1))

  node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='Plasma',
        size=20,
        colorbar=dict(
            thickness=15,
            title='Number of Connections',
            xanchor='left',
            tickmode='array',
            tickvals=tickvals
        )
    )
  )

  # Add hover text for nodes
  node_text = [f'{node}<br>Connections: {G.degree(node)}' for node in G.nodes()]
  node_trace.marker.color = [G.degree(node) for node in G.nodes()]
  node_trace.text = node_text

  # Create the figure and add edge and node traces
  fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Related Slang Network",
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white'
                ))

  graph_html = pio.to_html(fig, full_html=False)

  return graph_html

def scatter_pca(pca, slang, n=None):
  if n is not None:
      indices = np.random.choice(len(pca), n, replace=False)
      pca = pca[indices]
      slang = np.array(slang)[indices]

  source = ColumnDataSource(data=dict(
    x=pca[:, 0],
    y=pca[:, 1],
    slang=slang
  ))

  p = figure(title="PCA of FastText Word Embeddings",
           x_axis_label='Principal Component 1',
           y_axis_label='Principal Component 2',
           tools="pan,box_zoom,reset,hover")

  # Add scatter points
  p.scatter(x='x', y='y', source=source, size=10, alpha=0.6)

  # Add hover tool
  p.add_tools(
    HoverTool(
        tooltips=[("Slang", "@slang")]
    )
  )

  script, div = components(p)
  
  return script, div

def scatter_pca_with_clustering(pca, slang, n_clusters=5, num_points=None):
    if num_points is not None:
      indices = np.random.choice(len(pca), num_points, replace=False)
      pca = pca[indices]
      slang = np.array(slang)[indices]

    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(pca)

    clusters_str = [str(c) for c in clusters]

    source = ColumnDataSource(data=dict(
        x=pca[:, 0],
        y=pca[:, 1],
        slang=slang,
        cluster=clusters_str
    ))

    p = figure(title="PCA of FastText Word Embeddings (with Clusters)",
               x_axis_label='Principal Component 1',
               y_axis_label='Principal Component 2',
               tools="pan,box_zoom,reset,hover")

    color_mapper = CategoricalColorMapper(factors=[str(i) for i in range(n_clusters)],
                                          palette=Category10[n_clusters])

    p.scatter(x='x', y='y', source=source, size=10, color={'field': 'cluster', 'transform': color_mapper}, alpha=0.6)

    # Add hover tool
    p.add_tools(HoverTool(tooltips=[("Slang", "@slang"), ("Cluster", "@cluster")]))

    script, div = components(p)
  
    return script, div

def sentiment_over_time_p(df):
    sentiments = ['positive', 'negative', 'neutral']
    time_options = ['day', 'month', 'year']
    metric_options = ['count', 'percent']

    # Prepare time-based columns
    df['day'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['year'] = pd.to_datetime(df['date'].dt.year.astype(str))

    # Store all data combinations
    data_dict = {}
    for time_option in time_options:
        col = time_option
        for metric in metric_options:
            def_group = df.groupby([col, 'd_sentiment']).size().unstack(fill_value=0).reset_index()
            ex_group = df.groupby([col, 'e_sentiment']).size().unstack(fill_value=0).reset_index()

            # Fill missing sentiments
            for sentiment in sentiments:
                if sentiment not in def_group.columns:
                    def_group[sentiment] = 0
                if sentiment not in ex_group.columns:
                    ex_group[sentiment] = 0

            # Percent mode
            if metric == 'percent':
                def_totals = def_group[sentiments].sum(axis=1)
                def_group[sentiments] = def_group[sentiments].div(def_totals, axis=0) * 100

                ex_totals = ex_group[sentiments].sum(axis=1)
                ex_group[sentiments] = ex_group[sentiments].div(ex_totals, axis=0) * 100

            data_dict[(col, metric)] = (def_group, ex_group)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Sentiment of Definitions", "Sentiment of Examples"),
        shared_yaxes=True
    )

    # Add all traces, but only make one set visible at a time
    visibility_flags = []
    trace_labels = []

    sentiment_colors = {'positive': px.colors.qualitative.Set1[0],
                        'negative': px.colors.qualitative.Set1[1],
                        'neutral': px.colors.qualitative.Set1[2]}

    for time_option in time_options:
        for metric in metric_options:
            def_group, ex_group = data_dict[(time_option, metric)]

            for col_data, label_prefix, subplot_col in zip(
                [def_group, ex_group],
                ['Definition', 'Example'],
                [1, 2]
            ):
                for sentiment in sentiments:
                    trace = go.Scatter(
                        x=col_data[time_option],
                        y=col_data[sentiment],
                        mode='lines+markers',
                        name=f'{sentiment.capitalize()} {label_prefix}',
                        visible=False,
                        line=dict(color=sentiment_colors[sentiment]),
                        hovertemplate=f'<br>{time_option.capitalize()}: %{{x|%Y-%m-%d}}<br>{"Percent" if metric == "percent" else "Count"}: %{{y:.2f}}{"%" if metric == "percent" else ""}',
                    )
                    fig.add_trace(trace, row=1, col=subplot_col)
                    trace_labels.append((time_option, metric))

    # Make default selection visible (e.g., year + count)
    default_option = ('year', 'count')
    for i, label in enumerate(trace_labels):
        if label == default_option:
            fig.data[i].visible = True

    # Dropdown menu logic
    def make_visibility(time_sel, metric_sel):
        return [
            label == (time_sel, metric_sel) for label in trace_labels
        ]

    # Create dropdowns
    fig.update_layout(
        updatemenus=[
            # Time toggle
            {
                "buttons": [
                    {"label": time.capitalize(),
                     "method": "update",
                     "args": [{"visible": make_visibility(time, default_option[1])}]}
                    for time in time_options
                ],
                "direction": "up",
                "showactive": True,
                "x": 0.2,
                "y": -0.25,
                "xanchor": "left",
                "yanchor": "top",
            },
            # Count/Percent toggle
            {
                "buttons": [
                    {"label": "Count",
                     "method": "update",
                     "args": [{"visible": make_visibility(default_option[0], 'count')}]},
                    {"label": "Percent",
                     "method": "update",
                     "args": [{"visible": make_visibility(default_option[0], 'percent')}]}
                ],
                "direction": "up",
                "showactive": True,
                "x": 0,
                "y": -0.25,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        title="Sentiment of Slang Over Time",
        height=550,
        width=1000,
        hovermode="x unified",
        margin=dict(t=100),
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Count / Percent")

    graph_html = pio.to_html(fig, full_html=False)

    return graph_html

def emotion_over_time_p(df):
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy','optimism','sadness', 'surprise']
    time_options = ['day', 'month', 'year']
    metric_options = ['count', 'percent']

    # Prepare time-based columns
    df['day'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['year'] = pd.to_datetime(df['date'].dt.year.astype(str))

    # Store all data combinations
    data_dict = {}
    for time_option in time_options:
        col = time_option
        for metric in metric_options:
            def_group = df.groupby([col, 'd_emotion']).size().unstack(fill_value=0).reset_index()
            ex_group = df.groupby([col, 'e_emotion']).size().unstack(fill_value=0).reset_index()

            # Fill missing sentiments
            for emotion in emotions:
                if emotion not in def_group.columns:
                    def_group[emotion] = 0
                if emotion not in ex_group.columns:
                    ex_group[emotion] = 0

            # Percent mode
            if metric == 'percent':
                def_totals = def_group[emotions].sum(axis=1)
                def_group[emotions] = def_group[emotions].div(def_totals, axis=0) * 100

                ex_totals = ex_group[emotions].sum(axis=1)
                ex_group[emotions] = ex_group[emotions].div(ex_totals, axis=0) * 100

            data_dict[(col, metric)] = (def_group, ex_group)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Emotion of Definitions", "Emotion of Examples"),
        shared_yaxes=True
    )

    # Add all traces, but only make one set visible at a time
    visibility_flags = []
    trace_labels = []

    emotion_colors = {}
    for i, emotion in enumerate(emotions):
        emotion_colors[emotion] = px.colors.qualitative.Alphabet[i]

    for time_option in time_options:
        for metric in metric_options:
            def_group, ex_group = data_dict[(time_option, metric)]

            for col_data, label_prefix, subplot_col in zip(
                [def_group, ex_group],
                ['Definition', 'Example'],
                [1, 2]
            ):
                for emotion in emotions:
                    trace = go.Scatter(
                        x=col_data[time_option],
                        y=col_data[emotion],
                        mode='lines+markers',
                        name=f'{emotion.capitalize()} {label_prefix}',
                        visible=False,
                        line=dict(color=emotion_colors[emotion]),
                        hovertemplate=f'<br>{time_option.capitalize()}: %{{x|%Y-%m-%d}}<br>{"Percent" if metric == "percent" else "Count"}: %{{y:.2f}}{"%" if metric == "percent" else ""}',
                    )
                    fig.add_trace(trace, row=1, col=subplot_col)
                    trace_labels.append((time_option, metric))

    # Make default selection visible (e.g., year + count)
    default_option = ('year', 'count')
    for i, label in enumerate(trace_labels):
        if label == default_option:
            fig.data[i].visible = True

    # Dropdown menu logic
    def make_visibility(time_sel, metric_sel):
        return [
            label == (time_sel, metric_sel) for label in trace_labels
        ]

    # Create dropdowns
    fig.update_layout(
        updatemenus=[
            # Time toggle
            {
                "buttons": [
                    {"label": time.capitalize(),
                     "method": "update",
                     "args": [{"visible": make_visibility(time, default_option[1])}]}
                    for time in time_options
                ],
                "direction": "up",
                "showactive": True,
                "x": 0.2,
                "y": -0.25,
                "xanchor": "left",
                "yanchor": "top",
            },
            # Count/Percent toggle
            {
                "buttons": [
                    {"label": "Count",
                     "method": "update",
                     "args": [{"visible": make_visibility(default_option[0], 'count')}]},
                    {"label": "Percent",
                     "method": "update",
                     "args": [{"visible": make_visibility(default_option[0], 'percent')}]}
                ],
                "direction": "up",
                "showactive": True,
                "x": 0,
                "y": -0.25,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        title="Emotion of Slang Over Time",
        height=550,
        width=1000,
        hovermode="x unified",
        margin=dict(t=100),
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Count / Percent")

    graph_html = pio.to_html(fig, full_html=False)

    return graph_html

def get_linguistics_graphs(slang, non_slang):
  scripts = []
  divs = []
  s, d = linguistics.compare_alpha(slang, non_slang)
  scripts.append(s)
  divs.append(d)
  s, d = linguistics.first_char(slang)
  scripts.append(s)
  divs.append(d)
  s1, s2, d1, d2 = linguistics.compare_first_letter(slang, non_slang)
  scripts.append(s1)
  divs.append(d1)
  scripts.append(s2)
  divs.append(d2)
  s, d = linguistics.phoenetics(slang)
  scripts.append(s)
  divs.append(d)
  s1, s2, d1, d2 = linguistics.compare_phones(slang, non_slang)
  scripts.append(s1)
  divs.append(d1)
  scripts.append(s2)
  divs.append(d2)
  s, d = linguistics.num_words(slang)
  scripts.append(s)
  divs.append(d)
  s, d = linguistics.compare_dashes(slang, non_slang)
  scripts.append(s)
  divs.append(d)

  return scripts, divs

def get_social_graphs(urban_dict_data, slang, pca):
  scripts = []
  divs = []
  graph = related_network_graph(urban_dict_data, n=50)

  s, d = social.most_popular(urban_dict_data, slang)
  scripts.append(s)
  divs.append(d)
  s, d = social.num_entries(urban_dict_data, slang)
  scripts.append(s)
  divs.append(d)
  s1, s2, d1, d2 = social.curse_words(urban_dict_data)
  scripts.append(s1)
  divs.append(d1)
  scripts.append(s2)
  divs.append(d2)
  s, d = social.compare_curse_words(urban_dict_data)
  scripts.append(s)
  divs.append(d)
  s, d = social.related(urban_dict_data)
  scripts.append(s)
  divs.append(d)

  #s, d = scatter_pca(pca, slang)
  #scripts.append(s)
  #divs.append(d)
  #s, d = scatter_pca(pca, slang, n=10)
  #scripts.append(s)
  #divs.append(d)
  #s, d = scatter_pca(pca, slang, n=50)
  #scripts.append(s)
  #divs.append(d)
  #s, d = scatter_pca(pca, slang, n=100)
  #scripts.append(s)
  #divs.append(d)
  #s, d = scatter_pca_with_clustering(pca, slang)
  #scripts.append(s)
  #divs.append(d)
  #s, d = scatter_pca_with_clustering(pca, slang, num_points=1000)
  #scripts.append(s)
  #divs.append(d)
  return scripts, divs, graph

def get_sentimentemotion_graphs(urban_dict_data):
  scripts = []
  divs = []
  s, d = sentimentemotion.sentiment_counts(urban_dict_data)
  scripts.append(s)
  divs.append(d)
  s, d = sentimentemotion.emotion_counts(urban_dict_data)
  scripts.append(s)
  divs.append(d)
  s, d = sentimentemotion.emotion_cats(urban_dict_data)
  scripts.append(s)
  divs.append(d)
  
  return scripts, divs

def get_trends_graphs(dates_df, dates2_df, trends_df):
  scripts = []
  divs = []
  graphs = {}

  scripts = []
  divs = []
  s, d = trends.year(dates_df)
  scripts.append(s)
  divs.append(d)
  s, d = trends.date_info(dates_df, day=True)
  scripts.append(s)
  divs.append(d)
  s, d = trends.date_info(dates_df, month=True)
  scripts.append(s)
  divs.append(d)
  s, d = trends.date_info(dates_df, year=True)
  scripts.append(s)
  divs.append(d)
  s, d = trends.sent_stacked(dates2_df)
  scripts.append(s)
  divs.append(d)
  s, d = trends.emo_stacked(dates2_df)
  scripts.append(s)
  divs.append(d)
  s, d = trends.sent_line(dates2_df)
  scripts.append(s)
  divs.append(d)
  s, d = trends.emo_line(dates2_df)
  scripts.append(s)
  divs.append(d)
  s1, s2, d1, d2 = trends.sent_heatmap(dates2_df)
  scripts.append(s1)
  divs.append(d1)
  scripts.append(s2)
  divs.append(d2)
  s1, s2, d1, d2 = trends.emo_heatmap(dates2_df)
  scripts.append(s1)
  divs.append(d1)
  scripts.append(s2)
  divs.append(d2)

  graphs['graph1'] = sentiment_over_time_p(dates2_df)
  graphs['graph2'] = emotion_over_time_p(dates2_df)
  return scripts, divs, graphs

def get_data(info, definition=True):
  sent_to_num = {'positive': 1, 'negative': -1, 'neutral':0}
  emo_to_num = {'anger': 0, 'anticipation': 4, 'disgust':3, 'fear':1, 'joy':7, 'optimism':6, 'sadness':2, 'surprise':5 }
  data = {'date': [], 'sentiment': [], 'emotion': []}
  for entry in info['top_5_entries']:
    if definition:
      data['date'].append(entry['date'][:10])
      s = entry['definition_sentiment_label']
      data['sentiment'].append(sent_to_num[s])
      e = entry['definition_emotion_label']
      data['emotion'].append(emo_to_num[e])
    else:
      data['date'].append(entry['date'][:10])
      s = entry['example_sentiment_label']
      data['sentiment'].append(sent_to_num[s])
      e = entry['example_emotion_label']
      data['emotion'].append(emo_to_num[e])
  return data

def get_search_graphs(info):
  scripts = []
  divs = []
  data = get_data(info, True)
  s, d = search.sent_tracker(data, "(Definitions)")
  scripts.append(s)
  divs.append(d)
  data = get_data(info, False)
  scripts.append(s)
  divs.append(d)
  s, d = search.sent_tracker(data, "(Examples)")
  scripts.append(s)
  divs.append(d)
  data = get_data(info, True)
  s, d = search.emo_tracker(data, "(Definitions)")
  scripts.append(s)
  divs.append(d)
  data = get_data(info, True)
  s, d = search.emo_tracker(data, "(Examples)")
  scripts.append(s)
  divs.append(d)

  return scripts, divs