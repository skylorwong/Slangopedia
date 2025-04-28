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

def sentiment_counts(urban_dict):
  def_positive = 0
  def_negative = 0
  def_neutral = 0

  ex_positive = 0
  ex_negative = 0
  ex_neutral = 0

  for word in urban_dict:
    top_5 = urban_dict[word]['top_5_entries']
    for t in top_5:
      def_sentiment_label = t['definition_sentiment_label']
      if def_sentiment_label == 'positive':
        def_positive += 1
      elif def_sentiment_label == 'negative':
        def_negative += 1
      else:
        def_neutral += 1
      ex_sentiment_label = t['example_sentiment_label']
      if ex_sentiment_label == 'positive':
        ex_positive += 1
      elif ex_sentiment_label == 'negative':
        ex_negative += 1
      else:
        ex_neutral += 1

  labels = ['positive', 'negative', 'neutral']
  def_counts = [def_positive, def_negative, def_neutral]
  ex_counts = [ex_positive, ex_negative, ex_neutral]
  colors = ['green', 'red', 'gray']

  # Labels for x-axis
  sentiments = ['positive', 'negative', 'neutral']
  categories = ['Definition', 'Example']
  x = [(s, c) for s in sentiments for c in categories]

  counts = []
  for i in range(3):
    counts.append(def_counts[i])
    counts.append(ex_counts[i])

  palette = ['#718dbf', '#e84d60']  # Blue for Definition, Red for Example
  palette = Category10[3]

  # COMPARE SENTIMENT BY PERCENT
  total = np.sum(counts)/2
  percents = []
  for c in counts:
    percents.append(c/total*100)

  palette = ['#718dbf', '#e84d60']  # Blue for Definition, Red for Example
  palette = Category10[3]

  source = ColumnDataSource(data=dict(x=x, counts=percents,category=categories*len(sentiments)))

  p = figure(x_range=FactorRange(*x),
           height=400,
           title="Comparison of Sentiment in Definitions and Examples by Percent")

  p.vbar(x='x', top='counts', width=0.8, source=source,
       fill_color=factor_cmap('x', palette=palette, factors=categories, start=1), legend_field='category')

  # Add aesthetics
  p.y_range.start = 0
  p.x_range.range_padding = 0.05
  p.xaxis.major_label_text_font_size = '0pt'
  p.xgrid.grid_line_color = None
  p.xaxis.axis_label = "Sentiment"
  p.yaxis.axis_label = "Percent"

  p.legend.location = 'top_left'

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Sentiment", "@x"), ("Percent", "@counts")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def emotion_counts(urban_dict):
  def_anger = 0
  def_anticipation = 0
  def_disgust = 0
  def_fear = 0
  def_joy = 0
  def_optimism = 0
  def_sadness = 0
  def_surprise = 0

  ex_anger = 0
  ex_anticipation = 0
  ex_disgust = 0
  ex_fear = 0
  ex_joy = 0
  ex_optimism = 0
  ex_sadness =0
  ex_surprise = 0

  for word in urban_dict:
    top_5 = urban_dict[word]['top_5_entries']
    for t in top_5:
      def_emotion_label = t['definition_emotion_label']
      if def_emotion_label == 'anger':
        def_anger += 1
      elif def_emotion_label == 'anticipation':
        def_anticipation += 1
      elif def_emotion_label == 'disgust':
        def_disgust += 1
      elif def_emotion_label == 'fear':
        def_fear += 1
      elif def_emotion_label == 'joy':
        def_joy += 1
      elif def_emotion_label == 'optimism':
        def_optimism += 1
      elif def_emotion_label == 'sadness':
        def_sadness += 1
      elif def_emotion_label == 'surprise':
        def_surprise += 1
      else:
        continue

      ex_emotion_label = t['example_emotion_label']
      if ex_emotion_label == 'anger':
        ex_anger += 1
      elif ex_emotion_label == 'anticipation':
        ex_anticipation += 1
      elif ex_emotion_label == 'disgust':
        ex_disgust += 1
      elif ex_emotion_label == 'fear':
        ex_fear += 1
      elif ex_emotion_label == 'joy':
        ex_joy += 1
      elif ex_emotion_label == 'optimism':
        ex_optimism += 1
      elif ex_emotion_label == 'sadness':
        ex_sadness += 1
      elif ex_emotion_label == 'surprise':
        ex_surprise += 1
      else:
        continue

  labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy','optimism', 'sadness', 'surprise']
  def_counts = [def_anger, def_anticipation, def_disgust, def_fear, def_joy, def_optimism, def_sadness, def_surprise]
  ex_counts = [ex_anger, ex_anticipation, ex_disgust, ex_fear, ex_joy,ex_optimism, ex_sadness, ex_surprise]

  # DEFINITION EMOTION
  def_data = list(zip(labels, def_counts))
  def_data.sort(key=lambda x: x[1], reverse=True)
  sorted_labels, sorted_def_counts = zip(*def_data)

  # EXAMPLE EMOTION
  ex_data = list(zip(labels, ex_counts))
  ex_data.sort(key=lambda x: x[1], reverse=True)
  sorted_labels, sorted_ex_counts = zip(*ex_data)

  # Total count per emotion
  emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'optimism','sadness', 'surprise']
  total_counts = [d + e for d, e in zip(def_counts, ex_counts)]
  combined_data = list(zip(emotions, def_counts, ex_counts, total_counts))
  combined_data.sort(key=lambda x: x[3], reverse=True)  # sort by total

  # Extract sorted values
  sorted_emotions = [x[0] for x in combined_data]
  sorted_def_counts = [x[1] for x in combined_data]
  sorted_ex_counts = [x[2] for x in combined_data]

  categories = ['Definition', 'Example']
  x = [(e, c) for e in sorted_emotions for c in categories]

  counts = []
  for i in range(8):
    counts.append(sorted_def_counts[i])
    counts.append(sorted_ex_counts[i])

  # COMPARE EMOTION BY PERCENT
  total = np.sum(counts)/2
  percents = []
  for c in counts:
    percents.append(c/total*100)

  palette = Category10[3]

  source = ColumnDataSource(data=dict(x=x, counts=percents, category=categories*8))

  p = figure(x_range=FactorRange(*x),
           height=400,
           width=1000,
           title="Comparison of Emotion in Definitions and Examples by Percent")

  p.vbar(x='x', top='counts', width=0.8, source=source,
       fill_color=factor_cmap('x', palette=palette, factors=categories, start=1), legend_field='category')

  # Add aesthetics
  p.y_range.start = 0
  p.x_range.range_padding = 0.05
  p.xaxis.major_label_text_font_size = '0pt'
  p.xaxis.major_label_orientation = "vertical"
  p.xgrid.grid_line_color = None
  p.xaxis.axis_label = "Emotion"
  p.yaxis.axis_label = "Percent"

  p.legend.location = 'top_right'

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Emotion", "@x"), ("Percent", "@counts")]
  p.add_tools(hover)
  
  s, d = components(p)
  
  return s, d

def emotion_cats(urban_dict):
  def_anger = 0
  def_anticipation = 0
  def_disgust = 0
  def_fear = 0
  def_joy = 0
  def_love = 0
  def_optimism = 0
  def_pessimism = 0
  def_sadness = 0
  def_surprise = 0
  def_trust = 0

  ex_anger = 0
  ex_anticipation = 0
  ex_disgust = 0
  ex_fear = 0
  ex_joy = 0
  ex_love = 0
  ex_optimism = 0
  ex_pessimism = 0
  ex_sadness =0
  ex_surprise = 0
  ex_trust = 0

  for word in urban_dict:
    top_5 = urban_dict[word]['top_5_entries']
    for t in top_5:
      def_emotion_label = t['definition_emotion_label']
      if def_emotion_label == 'anger':
        def_anger += 1
      elif def_emotion_label == 'anticipation':
        def_anticipation += 1
      elif def_emotion_label == 'disgust':
        def_disgust += 1
      elif def_emotion_label == 'fear':
        def_fear += 1
      elif def_emotion_label == 'joy':
        def_joy += 1
      elif def_emotion_label == 'love':
        def_love += 1
      elif def_emotion_label == 'optimism':
        def_optimism += 1
      elif def_emotion_label == 'pessimism':
        def_pessimism += 1
      elif def_emotion_label == 'sadness':
        def_sadness += 1
      elif def_emotion_label == 'surprise':
        def_surprise += 1
      else:
        def_trust += 1

      ex_emotion_label = t['example_emotion_label']
      if ex_emotion_label == 'anger':
        ex_anger += 1
      elif ex_emotion_label == 'anticipation':
        ex_anticipation += 1
      elif ex_emotion_label == 'disgust':
        ex_disgust += 1
      elif ex_emotion_label == 'fear':
        ex_fear += 1
      elif ex_emotion_label == 'joy':
        ex_joy += 1
      elif ex_emotion_label == 'love':
        ex_love += 1
      elif ex_emotion_label == 'optimism':
        ex_optimism += 1
      elif ex_emotion_label == 'pessimism':
        ex_pessimism += 1
      elif ex_emotion_label == 'sadness':
        ex_sadness += 1
      elif ex_emotion_label == 'surprise':
        ex_surprise += 1
      else:
        ex_trust += 1

  labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love','optimism','pessimism', 'sadness', 'surprise', 'trust']
  def_counts = [def_anger, def_anticipation, def_disgust, def_fear, def_joy, def_love, def_optimism, def_pessimism, def_sadness, def_surprise, def_trust]
  ex_counts = [ex_anger, ex_anticipation, ex_disgust, ex_fear, ex_joy, ex_love, ex_optimism, ex_pessimism, ex_sadness, ex_surprise, ex_trust]

  # DEFINITION EMOTION
  def_data = list(zip(labels, def_counts))
  def_data.sort(key=lambda x: x[1], reverse=True)
  sorted_labels, sorted_def_counts = zip(*def_data)

  # EXAMPLE EMOTION
  ex_data = list(zip(labels, ex_counts))
  ex_data.sort(key=lambda x: x[1], reverse=True)
  sorted_labels, sorted_ex_counts = zip(*ex_data)

  # Total count per emotion
  emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love','optimism','pessimism', 'sadness', 'surprise', 'trust']
  total_counts = [d + e for d, e in zip(def_counts, ex_counts)]
  combined_data = list(zip(emotions, def_counts, ex_counts, total_counts))
  combined_data.sort(key=lambda x: x[3], reverse=True)  # sort by total

  # Extract sorted values
  sorted_emotions = [x[0] for x in combined_data]
  sorted_def_counts = [x[1] for x in combined_data]
  sorted_ex_counts = [x[2] for x in combined_data]

  categories = ['Definition', 'Example']
  x = [(e, c) for e in sorted_emotions for c in categories]

  counts = []
  for i in range(11):
    counts.append(sorted_def_counts[i])
    counts.append(sorted_ex_counts[i])

  # COMPARE EMOTION BY PERCENT
  total = np.sum(counts)/2
  percents = []
  for c in counts:
    percents.append(c/total*100)

  palette = Category10[3]

  # POSITIVE VS NEGATIVE
  filtered_x = x[:-6]
  filtered_percents = percents[:-6]

  positive = ['joy', 'optimism']
  negative = ['anger', 'disgust', 'fear', 'sadness']
  neutral = ['anticipation', 'surprise']

  p_sentiment = []
  n_sentiment = []
  nn_sentiment = []
  labels = ['positive', 'negative', 'neutral']
  for i, l in enumerate(filtered_x):
    if l[0] in positive:
      p_sentiment.append(filtered_percents[i])
    elif l[0] in negative:
      n_sentiment.append(filtered_percents[i])
    else:
      nn_sentiment.append(filtered_percents[i])
  total = np.sum(p_sentiment) + np.sum(n_sentiment) + np.sum(nn_sentiment)
  p_total = np.sum(p_sentiment)/total*100
  n_total = np.sum(n_sentiment)/total*100
  nn_total = np.sum(nn_sentiment)/total*100

  p = figure(x_range=labels,\
             title=f"Emotion Categories of Slang",\
             x_axis_label='Emotion Category',\
             y_axis_label='Percent')

  # Create vertical bars
  p.vbar(x=labels, top=[p_total,n_total, nn_total], width=0.5, fill_color=list(Category10[3]))

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Emotion Category", "@x"), ("Percent", "@top")]
  p.add_tools(hover)

  s, d = components(p)
  
  return s, d