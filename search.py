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

def sent_tracker(data, definition):
  num_to_sent = {1: 'positive', -1: 'negative', 0:'neutral'}
  sent_to_colors = {1: Category10[3][0], -1: Category10[3][1], 0: Category10[3][2]}
  df = pd.DataFrame(data)
  df = df.sort_values(by='date')

  dates = [str(d) for d in df['date']]
  sentiments = list(df['sentiment'])
  title = f"Sentiment Tracker {definition}"
  colors = [sent_to_colors[v] for v in df['sentiment']]
  labels = [num_to_sent[v] for v in df['sentiment']]

  source = ColumnDataSource(data=dict(
    x=dates,
    y=sentiments,
    labels=labels,
    colors=colors
  ))

  p = figure(x_range=dates,\
             title=title,\
             x_axis_label='Date',\
             y_axis_label='Sentiment')

  # Create vertical bars
  p.scatter(x='x', y='y', color='colors', size=15, source=source)

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Date", "@x"), ("Sentiment", "@labels")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def emo_tracker(data, definition):
  num_to_emo = {0: 'anger', 4: 'anticipation', 3:'disgust', 1:'fear', 7:'joy', 6:'optimism', 2:'sadness', 5:'surprise' }
  emo_to_colors = {0: Category10[8][0], 1: Category10[8][1], 2: Category10[8][2], 3: Category10[8][3],4: Category10[8][4],5: Category10[8][5],6: Category10[8][6],7: Category10[8][7]}
  df = pd.DataFrame(data)
  df = df.sort_values(by='date')

  dates = [str(d) for d in df['date']]
  emotions = list(df['emotion'])
  title = f"Emotion Tracker {definition}"
  colors = [emo_to_colors[v] for v in df['emotion']]
  labels = [num_to_emo[v] for v in df['emotion']]

  source = ColumnDataSource(data=dict(
    x=dates,
    y=emotions,
    labels=labels,
    colors=colors
  ))

  p = figure(x_range=dates,\
             title=title,\
             x_axis_label='Date',\
             y_axis_label='Emotion')

  # Create vertical bars
  p.scatter(x='x', y='y', color='colors', size=15, source=source)

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Date", "@x"), ("Emotion", "@labels")]
  p.add_tools(hover)

  # Show the plot
  script, div = components(p)
  
  return script, div

def popularity_tracker(popularity):
  df = pd.DataFrame(popularity)
  df = df.sort_values(by='dates')

  dates = [str(d) for d in df['dates']]
  thumbs_up = list(df['thumbs'])

  source = ColumnDataSource(data=dict(x=dates,y=thumbs_up))

  p = figure(x_range=dates,\
             title="Popularity Tracker",\
             x_axis_label='Date',\
             y_axis_label='Number of Thumbs Up')

  # Create vertical bars
  p.vbar(x='x', top='y', width=0.5, color=Category10[3][0], source=source)

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Date", "@x"), ("Number of Thumbs Up", "@top")]
  p.add_tools(hover)
  
  script, div = components(p)
  
  return script, div