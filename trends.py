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

def year(df):
  word_counts = df.groupby('year_datetime').size().reset_index(name='count')
  word_counts['year_datetime'] = word_counts['year_datetime'].astype(str).str[:4]

  source = ColumnDataSource(word_counts)

  p = figure(x_range=word_counts['year_datetime'], height=400, width=800,
           title=f"Number of Slang by Year")

  p.vbar(x='year_datetime', top='count', width=0.8, source=source, color=Category10[3][0])

  p.xaxis.axis_label = "Year"
  p.yaxis.axis_label = "Count"
  p.xgrid.grid_line_color = None
  p.xaxis.major_label_orientation = 45
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [
        ("Year", "@year_datetime"),
        ("Count", "@count")
  ]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def date_info(df, day=False, month=False, year=False):
  if day:
    x = 'date'
    x_label = 'Day'
  elif month:
    x = 'month_datetime'
    x_label = 'Month'
  elif year:
    x = 'year_datetime'
    x_label = 'Year'
  else:
    print("Select day, month, or year")
    return

  word_counts = df.groupby(x).size().reset_index(name='count')
  source = ColumnDataSource(word_counts)

  p = figure(x_axis_type='datetime', x_axis_label=x_label, y_axis_label="Count", title=f"Number of Slang by {x_label}", height=400, width=800)
  p.line(x=x, y='count', source=source, line_width=2, color=Category10[3][0])

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [
        (x_label, f"@{x}{{%F}}"),
        ("Count", "@count")
    ]
  hover.formatters = {f'@{x}': 'datetime'}
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def sent_stacked(df):
  df['year_datetime'] = pd.to_datetime(df['year'], format='%Y')
  col = 'year_datetime'
  col_label = 'Year'

  # DEFINITION SENTIMENT
  sentiment_percentages = df.groupby([col, 'd_sentiment']).size().unstack(fill_value=0).reset_index()

  row_totals = sentiment_percentages[['negative', 'neutral', 'positive']].sum(axis=1)
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].div(row_totals, axis=0) * 100
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].round(2)

  # EXAMPLE SENTIMENT
  sentiment_percentages = df.groupby([col, 'e_sentiment']).size().unstack(fill_value=0).reset_index()

  row_totals = sentiment_percentages[['negative', 'neutral', 'positive']].sum(axis=1)
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].div(row_totals, axis=0) * 100
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].round(2)

  colors = Category10[3]
  source = ColumnDataSource(sentiment_percentages)
  sentiments = ['positive', 'negative', 'neutral']

  # COMBINED
  df_e = df[['year', 'e_sentiment']].rename(columns={'e_sentiment': 'sentiment'})
  df_d = df[['year', 'd_sentiment']].rename(columns={'d_sentiment': 'sentiment'})

  # Step 2: Combine the two DataFrames
  df_combined = pd.concat([df_e, df_d], ignore_index=True)
  col='year'
  col_label = 'Year'

  sentiment_percentages = df_combined.groupby(['year', 'sentiment']).size().unstack(fill_value=0).reset_index()

  row_totals = sentiment_percentages[['negative', 'neutral', 'positive']].sum(axis=1)
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].div(row_totals, axis=0) * 100
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].round(2)

  colors = Category10[3]
  source = ColumnDataSource(sentiment_percentages)
  sentiments = ['positive', 'negative', 'neutral']

  p = figure(title="Percent Of Each Sentiment Per Year (Definitions and Examples Combined)", height=400, width=800)
  p.varea_stack(stackers=sentiments, x=col, color=colors, legend_label=sentiments, source=source)

  p.xaxis.axis_label = "Year"
  p.yaxis.axis_label = "Percent"
  p.legend.location = "top_left"

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [
        (col_label, "@year"),
        ("Positive %", "@positive"),
        ("Negative %", "@negative"),
        ("Neutral %", "@neutral")
    ]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def emo_stacked(df):
  df['year_datetime'] = pd.to_datetime(df['year'], format='%Y')
  col = 'year_datetime'
  col_label = 'Year'

  colors = Category20[8]
  emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'optimism', 'sadness', 'surprise']

  # DEFINITION EMOTION
  emotion_percentages = df.groupby(['year_datetime', 'd_emotion']).size().unstack(fill_value=0).reset_index()

  available_emotions = [e for e in emotions if e in emotion_percentages.columns]
  row_totals = emotion_percentages[available_emotions].sum(axis=1)
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].div(row_totals, axis=0) * 100
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].round(2)

  # EXAMPLE EMOTION
  emotion_percentages = df.groupby(['year_datetime', 'e_emotion']).size().unstack(fill_value=0).reset_index()

  available_emotions = [e for e in emotions if e in emotion_percentages.columns]
  row_totals = emotion_percentages[available_emotions].sum(axis=1)
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].div(row_totals, axis=0) * 100
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].round(2)

  # COMBINED
  df_e = df[['year', 'e_emotion']].rename(columns={'e_emotion': 'emotion'})
  df_d = df[['year', 'd_emotion']].rename(columns={'d_emotion': 'emotion'})

  # Step 2: Combine the two DataFrames
  df_combined = pd.concat([df_e, df_d], ignore_index=True)

  emotion_percentages = df_combined.groupby(['year', 'emotion']).size().unstack(fill_value=0).reset_index()

  available_emotions = [e for e in emotions if e in emotion_percentages.columns]
  row_totals = emotion_percentages[available_emotions].sum(axis=1)
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].div(row_totals, axis=0) * 100
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].round(2)

  source = ColumnDataSource(emotion_percentages)
  col='year'
  col_label = 'Year'

  p = figure(title="Percent of Each Emotion Per Year (Defintions and Examples Combined)", height=400, width=800)
  p.varea_stack(stackers=emotions, x=col, color=colors, legend_label=emotions, source=source)

  p.yaxis.axis_label = "Percent"
  p.legend.location = "top_left"

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [
        (col_label, "@year"),
        ("Anger %", "@anger"),
        ("Anticipation %", "@anticipation"),
        ("Disgust %", "@disgust"),
        ("Fear %", "@fear"),
        ("Joy %", "@joy"),
        ("Optimism %", "@optimism"),
        ("Sadness %", "@sadness"),
        ("Surprise %", "@surprise")
    ]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def sent_line(df):
    # COMBINED
    df_e = df[['year', 'e_sentiment']].rename(columns={'e_sentiment': 'sentiment'})
    df_d = df[['year', 'd_sentiment']].rename(columns={'d_sentiment': 'sentiment'})

    # Step 2: Combine the two DataFrames
    df_combined = pd.concat([df_e, df_d], ignore_index=True)

    sentiment_percentages = df_combined.groupby(['year', 'sentiment']).size().unstack(fill_value=0).reset_index()

    row_totals = sentiment_percentages[['negative', 'neutral', 'positive']].sum(axis=1)
    sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].div(row_totals, axis=0) * 100
    sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].round(2)
    colors = Category10[3]

    source = ColumnDataSource(sentiment_percentages)
    sentiments = ['positive', 'negative', 'neutral']
    #colors = ['green', 'red', 'gray']

    p = figure(x_axis_label='Year', y_axis_label="Percent", title=f"Sentiment of Slang Over Time (Definitions and Examples Combined)", height=400, width=800)

    sentiments = ['positive', 'negative', 'neutral']
    for i, sentiment in enumerate(sentiments):
      p.line(x='year', y=sentiment, source=source, line_width=2, color=colors[i], legend_label=sentiment)

    # Customize the plot appearance
    p.yaxis.axis_label = "Percent"
    p.legend.location = "top_right"

    # Hover tool
    hover = HoverTool()
    hover.tooltips = [
        ("Year", "@year"),
        ("Positive Percent", "@positive"),
        ("Negative Percent", "@negative"),
        ("Neutral Percent", "@neutral")
      ]
    hover.formatters = {f'@year': 'datetime'}
    p.add_tools(hover)

    script, div = components(p)
  
    return script, div

def emo_line(df):
  emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'optimism', 'sadness', 'surprise']

  colors = Category10[8]

  # COMBINED
  df_e = df[['year', 'e_emotion']].rename(columns={'e_emotion': 'emotion'})
  df_d = df[['year', 'd_emotion']].rename(columns={'d_emotion': 'emotion'})

  # Step 2: Combine the two DataFrames
  df_combined = pd.concat([df_e, df_d], ignore_index=True)

  e_percentages = df_combined.groupby(['year', 'emotion']).size().unstack(fill_value=0).reset_index()

  row_totals = e_percentages[emotions].sum(axis=1)
  e_percentages[emotions] = e_percentages[emotions].div(row_totals, axis=0) * 100
  e_percentages[emotions] = e_percentages[emotions].round(2)

  source = ColumnDataSource(e_percentages)

  p = figure(x_axis_label="Year", y_axis_label="Percent", title="Emotions of Slang Over Time (Examples and Definitions Combined)", height=400, width=800)

  for i, emotion in enumerate(emotions):
    p.line(x='year', y=emotion, source=source, line_width=2, color=colors[i], legend_label=emotion)

  # Customize the plot appearance
  p.yaxis.axis_label = "Percent"
  p.legend.location = "top_left"

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [
        ('Year', "@year"),
        ("Anger Count", "@anger"),
        ("Anticipation Count", "@anticipation"),
        ("Disgust Count", "@disgust"),
        ("Fear Count", "@fear"),
        ("Joy Count", "@joy"),
        ("Optimism Count", "@optimism"),
        ("Sadness Count", "@sadness"),
        ("Surprise Count", "@surprise")
    ]
  hover.formatters = {"@year": 'datetime'}
  p.add_tools(hover)
  
  script, div = components(p)
  return script, div

def sent_heatmap(df):
  # DEFINITION SENTIMENT
  sentiment_percentages = df.groupby(['year', 'd_sentiment']).size().unstack(fill_value=0).reset_index()

  row_totals = sentiment_percentages[['negative', 'neutral', 'positive']].sum(axis=1)
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].div(row_totals, axis=0) * 100
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].round(2)

  years = sentiment_percentages['year'].astype(str).tolist()
  positive = sentiment_percentages['positive'].tolist()
  negative = sentiment_percentages['negative'].tolist()
  neutral = sentiment_percentages['neutral'].tolist()

  # Prepare heatmap data
  x = years * 3
  y = ['positive'] * len(years) + ['negative'] * len(years) + ['neutral'] * len(years)
  value = positive + negative + neutral

  heatmap_df = pd.DataFrame({'x': x, 'y': y, 'value': value})
  source = ColumnDataSource(heatmap_df)

  # Color mapping
  mapper = linear_cmap(field_name='value', palette=Viridis256, low=min(value), high=max(value))

  # Create figure with categorical axes
  p = figure(title="Sentiment Heaptmap By Percent (Definitions)",
           x_range=sorted(list(set(x))),
           y_range=['positive', 'negative', 'neutral'],
           x_axis_location="above",
           width=800, height=400,
           tools="hover", tooltips=[("Year", "@x"), ("Sentiment", "@y"), ("%", "@value")])

  # Add rectangles for heatmap
  p.rect(x="x", y="y", width=1, height=1, source=source,
       fill_color=mapper, line_color=None)

  # Add color bar
  color_bar = ColorBar(color_mapper=mapper['transform'], location=(0, 0))
  p.add_layout(color_bar, 'right')

  p.xaxis.major_label_orientation = 45

  script, div = components(p)

  # EXAMPLE SENTIMENT
  sentiment_percentages = df.groupby(['year', 'e_sentiment']).size().unstack(fill_value=0).reset_index()

  row_totals = sentiment_percentages[['negative', 'neutral', 'positive']].sum(axis=1)
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].div(row_totals, axis=0) * 100
  sentiment_percentages[['negative', 'neutral', 'positive']] = sentiment_percentages[['negative', 'neutral', 'positive']].round(2)

  years = sentiment_percentages['year'].astype(str).tolist()
  positive = sentiment_percentages['positive'].tolist()
  negative = sentiment_percentages['negative'].tolist()
  neutral = sentiment_percentages['neutral'].tolist()

  # Prepare heatmap data
  x = years * 3
  y = ['positive'] * len(years) + ['negative'] * len(years) + ['neutral'] * len(years)
  value = positive + negative + neutral

  heatmap_df = pd.DataFrame({'x': x, 'y': y, 'value': value})
  source = ColumnDataSource(heatmap_df)

  # Color mapping
  mapper = linear_cmap(field_name='value', palette=Viridis256, low=min(value), high=max(value))

  # Create figure with categorical axes
  p = figure(title="Sentiment Heaptmap By Percent (Examples)",
           x_range=sorted(list(set(x))),
           y_range=['positive', 'negative', 'neutral'],
           x_axis_location="above",
           width=800, height=400,
           tools="hover", tooltips=[("Year", "@x"), ("Sentiment", "@y"), ("%", "@value")])

  # Add rectangles for heatmap
  p.rect(x="x", y="y", width=1, height=1, source=source,
       fill_color=mapper, line_color=None)

  # Add color bar
  color_bar = ColorBar(color_mapper=mapper['transform'], location=(0, 0))
  p.add_layout(color_bar, 'right')

  p.xaxis.major_label_orientation = 45

  s2, d2 = components(p)

  return script, s2, div, d2

def emo_heatmap(df):
  emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'optimism','sadness', 'surprise']

  # DEFINITION EMOTION
  emotion_percentages = df.groupby(['year', 'd_emotion']).size().unstack(fill_value=0).reset_index()

  available_emotions = [e for e in emotions if e in emotion_percentages.columns]
  row_totals = emotion_percentages[available_emotions].sum(axis=1)
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].div(row_totals, axis=0) * 100
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].round(2)

  years = emotion_percentages['year'].astype(str).tolist()
  y = []
  value = []
  for emotion in emotions:
    y += [emotion] * len(years)
    if emotion in emotion_percentages:
      value += emotion_percentages[emotion].tolist()
    else:
      value += [0] * len(years)
  x = years * len(emotions)

  heatmap_df = pd.DataFrame({'x': x, 'y': y, 'value': value})
  source = ColumnDataSource(heatmap_df)

  # Color mapping
  mapper = linear_cmap(field_name='value', palette=Viridis256, low=min(value), high=max(value))

  # Create figure with categorical axes
  p = figure(title="Emotion Heaptmap By Percent Per Year (Definitions)",
           x_range=sorted(list(set(x))),
           y_range=emotions,
           x_axis_location="above",
           width=800, height=400,
           tools="hover", tooltips=[("Year", "@x"), ("Emotion", "@y"), ("%", "@value")])

  # Add rectangles for heatmap
  p.rect(x="x", y="y", width=1, height=1, source=source,
       fill_color=mapper, line_color=None)

  # Add color bar
  color_bar = ColorBar(color_mapper=mapper['transform'], location=(0, 0))
  p.add_layout(color_bar, 'right')

  p.xaxis.major_label_orientation = 45

  s1, d1 = components(p)

  # EXAMPLE EMOTION
  emotion_percentages = df.groupby(['year', 'e_emotion']).size().unstack(fill_value=0).reset_index()

  available_emotions = [e for e in emotions if e in emotion_percentages.columns]
  row_totals = emotion_percentages[available_emotions].sum(axis=1)
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].div(row_totals, axis=0) * 100
  emotion_percentages[available_emotions] = emotion_percentages[available_emotions].round(2)

  years = emotion_percentages['year'].astype(str).tolist()
  y = []
  value = []
  for emotion in emotions:
    y += [emotion] * len(years)
    if emotion in emotion_percentages:
      value += emotion_percentages[emotion].tolist()
    else:
      value += [0] * len(years)
  x = years * len(emotions)

  heatmap_df = pd.DataFrame({'x': x, 'y': y, 'value': value})
  source = ColumnDataSource(heatmap_df)

  # Color mapping
  mapper = linear_cmap(field_name='value', palette=Viridis256, low=min(value), high=max(value))

  # Create figure with categorical axes
  p = figure(title="Emotion Heaptmap By Percent (Examples)",
           x_range=sorted(list(set(x))),
           y_range=emotions,
           x_axis_location="above",
           width=800, height=400,
           tools="hover", tooltips=[("Year", "@x"), ("Emotion", "@y"), ("%", "@value")])

  # Add rectangles for heatmap
  p.rect(x="x", y="y", width=1, height=1, source=source,
       fill_color=mapper, line_color=None)

  # Add color bar
  color_bar = ColorBar(color_mapper=mapper['transform'], location=(0, 0))
  p.add_layout(color_bar, 'right')

  p.xaxis.major_label_orientation = 45
  s2, d2 = components(p)

  return s1, s2, d1, d2