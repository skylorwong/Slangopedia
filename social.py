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

def most_popular(urban_dict, slang, top=14):
  thumbs_up = []
  for s in slang:
    thumbs_up.append(urban_dict[s]['top_5_entries'][0]['thumbs_up'])

  indices_of_top = np.argsort(thumbs_up)[-top:][::-1]
  top_slang = np.array(slang)[indices_of_top]
  top_thumbs = np.array(thumbs_up)[indices_of_top]

  new_top_slang = []
  new_top_thumbs = []
  for i, s in enumerate(top_slang):
    if s in ["the slut", "woodie", "Nerd", "this is shit"]:
      continue
    else:
      new_top_slang.append(s)
      new_top_thumbs.append([top_thumbs[i]])

  top_slang = new_top_slang
  top_thumbs = new_top_thumbs

  p = figure(x_range=top_slang,\
             title=f"Top 10 Most Popular Slang",\
             x_axis_label='Slang',\
             y_axis_label='Number of Thumbs Up')

  # Create vertical bars
  p.vbar(x=top_slang, top=top_thumbs, width=0.9,color=Category10[3][0])

  # Customize plot aesthetics
  p.xaxis.major_label_orientation = 45
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Slang", "@x"), ("Number of Thumbs Up", "@top")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def num_entries(urban_dict, slang):
  entries_count = {}
  for s in slang:
    entries = urban_dict[s]['num_entries']
    entries_count[entries] = entries_count.get(entries, 0) + 1

  sorted_dict = dict(sorted(entries_count.items(), key=lambda item: item[1], reverse=True))

  entries = list(sorted_dict.keys())
  for i, entry in enumerate(entries):
    if entry != 10:
      entries[i] = str(entry)
    else:
      entries[i] = "10+"
  counts = list(sorted_dict.values())

  p = figure(x_range=entries,\
             title="Number of User Contributions Per Slang",\
             x_axis_label="Number of User Contributions",\
             y_axis_label='Frequency')

  # Create vertical bars
  p.vbar(x=entries, top=counts, width=0.9, color=Category10[3][0])

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Number of User Contributions", "@x"), ("Frequency", "@top")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def curse_words(urban_dict):
  curse_words = ["fuck",\
                 "shit",\
                 "bitch",\
                 "crap",\
                 "ass",\
                 "damn",\
                 ]

  curse_counts = {word: 0 for word in curse_words}

  ex_or_def = 'definition'
  for slang, info in urban_dict.items():
    for entry in info['top_5_entries']:
      text = entry[ex_or_def]
      for curse in curse_counts:
        count = text.lower().count(curse)
        curse_counts[curse] = curse_counts.get(curse, 0) + count

  sorted_dict = dict(sorted(curse_counts.items(), key=lambda item: item[1], reverse=True))

  curse = list(sorted_dict.keys())
  counts = list(sorted_dict.values())
  title = "Frequency of Curse Words in Definitions"

  percents = []
  total = len(urban_dict)
  for c in counts:
    percents.append(c/total*100)

  p = figure(x_range=curse,\
             title="Percent of Slang Definitions with Curse Words",\
             x_axis_label='Curse Words',\
             y_axis_label='Percent')

  # Create vertical bars
  p.vbar(x=curse, top=percents, width=0.9, color=Category10[3][0])

  # Customize plot aesthetics
  p.xaxis.major_label_orientation = 45
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Curse Word", "@x"), ("Percent", "@top")]
  p.add_tools(hover)

  s1, d1 = components(p)

  curse_counts = {word: 0 for word in curse_words}

  ex_or_def = 'example'
  for slang, info in urban_dict.items():
    for entry in info['top_5_entries']:
      text = entry[ex_or_def]
      for curse in curse_counts:
        count = text.lower().count(curse)
        curse_counts[curse] = curse_counts.get(curse, 0) + count

  sorted_dict = dict(sorted(curse_counts.items(), key=lambda item: item[1], reverse=True))

  curse = list(sorted_dict.keys())
  counts = list(sorted_dict.values())
  title = "Frequency of Curse Words in Examples"

  percents = []
  total = len(urban_dict)
  for c in counts:
    percents.append(c/total*100)

  p = figure(x_range=curse,\
             title="Percent of Slang Examples with Curse Words",\
             x_axis_label='Curse Words',\
             y_axis_label='Percent')

  # Create vertical bars
  p.vbar(x=curse, top=percents, width=0.9, color=Category10[3][0])

  # Customize plot aesthetics
  p.xaxis.major_label_orientation = 45
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Curse Word", "@x"), ("Percent", "@top")]
  p.add_tools(hover)

  s2, d2 = components(p)
  
  return s1, s2, d1, d2

def compare_curse_words(urban_dict):
  curse_words = ["fuck",\
                 "shit",\
                 "bitch",\
                 "crap",\
                 "ass",\
                 "damn",\
                 ]

  def_curse_counts = {word: 0 for word in curse_words}
  ex_curse_counts = {word: 0 for word in curse_words}

  for slang, info in urban_dict.items():
    for entry in info['top_5_entries']:
      definition = entry['definition']
      example = entry['example']
      for curse in curse_words:
        def_count = definition.lower().count(curse)
        ex_count = example.lower().count(curse)
        def_curse_counts[curse] = def_curse_counts.get(curse, 0) + def_count
        ex_curse_counts[curse] = ex_curse_counts.get(curse, 0) + ex_count

  # Labels for x-axis
  categories = ['Definition', 'Example']
  x = [(curse, category) for curse in curse_words for category in categories]

  counts = []
  for curse in curse_words:
    counts.append(def_curse_counts[curse])
    counts.append(ex_curse_counts[curse])

  palette = ['#718dbf', '#e84d60']  # Blue for Definition, Red for Example
  palette = Category10[3]

  source = ColumnDataSource(data=dict(x=x, counts=counts, category=categories*len(curse_words)))

  p = figure(x_range=FactorRange(*x),
           height=400,
           title="Comparison of Curse Word Counts in Definitions and Examples")

  p.vbar(x='x', top='counts', width=0.8, source=source,
       fill_color=factor_cmap('x', palette=palette, factors=categories, start=1), legend_field='category')

  # Add aesthetics
  p.y_range.start = 0
  p.xaxis.major_label_orientation = "vertical"
  p.xaxis.major_label_text_font_size = '0pt'
  p.x_range.range_padding = 0.05
  p.xgrid.grid_line_color = None
  p.xaxis.axis_label = "Curse Word"
  p.yaxis.axis_label = "Count"

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Curse Word", "@x"), ("Count", "@counts")]
  p.add_tools(hover)

  s, d = components(p)
  
  return s, d

def related(urban_dict):
  related_counts = {}

  for slang, info in urban_dict.items():
    for entry in info['top_5_entries']:
      num_related = len(entry['related'])
      if num_related >= 10:
        key = "10+"
      else:
        key = str(num_related)
      related_counts[key] = related_counts.get(key, 0) + 1

  sorted_dict = dict(sorted(related_counts.items(), key=lambda item: item[1], reverse=True))

  related = list(sorted_dict.keys())
  related = [str(r) for r in related]
  counts = list(sorted_dict.values())

  # PERCENT
  percents = []
  total = np.sum(counts)
  for c in counts:
    percents.append(c/total*100)
  p = figure(x_range=related,\
             title="Number of Related Slang Per Definition",\
             x_axis_label='Number of Related Slang',\
             y_axis_label='Percent')

  # Create vertical bars
  p.vbar(x=related, top=percents, width=0.5, color=Category10[3][0])

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Number of Related Slang", "@x"), ("Percent", "@top")]
  p.add_tools(hover)

  s, d = components(p)
  
  return s, d