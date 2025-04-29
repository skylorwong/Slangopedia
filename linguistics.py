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

def first_char(slang):
  first_chars = [s[0].upper() for s in slang]
  unique_chars, counts = np.unique(first_chars, return_counts=True)

  sorted_indices = np.argsort(counts)[::-1]  # Get indices that would sort counts in ascending order
  unique_chars_sorted = unique_chars[sorted_indices]
  counts_sorted = counts[sorted_indices]  

  p = figure(x_range=unique_chars_sorted,\
             title="Number of Slang Starting With Each Character",\
             x_axis_label='First Character',\
             y_axis_label='Count')

  # Create vertical bars
  p.vbar(x=unique_chars_sorted, top=counts_sorted, width=0.5, color=Category10[3][0])

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Character", "@x"), ("Count", "@top")]
  p.add_tools(hover)
  
  script, div = components(p)
  
  return script, div

def get_percents(words):
  num_alpha = 0
  num_non_alpha = 0
  for s in words:
    words = s.split()
    word_count = 0
    all_alpha = True
    for word in words:
      if not (word.isalpha() and word.isascii()):
        all_alpha = False
        break
    if all_alpha:
      num_alpha += 1
    else:
      num_non_alpha += 1

  percent_alpha = num_alpha/(num_alpha+ num_non_alpha)*100
  percent_non_alpha = num_non_alpha/(num_alpha+ num_non_alpha)*100

  return percent_alpha, percent_non_alpha

def compare_alpha(slang, non_slang):
  percent_s_alpha, percent_s_non_alpha = get_percents(slang)
  percent_ns_alpha, percent_ns_non_alpha = get_percents(non_slang)

  categories = ['Urban Dictionary', 'Standard English Dictionary']
  percents = [percent_s_alpha, percent_ns_alpha, percent_s_non_alpha, percent_ns_non_alpha]
  compare = ["Alphabetic", "Non-Alphabetic"]

  x = [(c, category) for c in compare for category in categories]

  palette = Category10[3]

  source = ColumnDataSource(data=dict(x=x, percents=percents, category=categories*2))

  p = figure(x_range=FactorRange(*x),
           height=400,
           title="Percent of Alphabetic vs Non-Alphabetic Words",
           tools="pan,box_zoom,reset,save")

  p.vbar(x='x', top='percents', width=0.8, source=source,
       fill_color=factor_cmap('x', palette=palette, factors=categories, start=1), legend_field='category')

  # Add aesthetics
  p.y_range.start = 0
  p.xaxis.major_label_orientation = "vertical"
  p.xaxis.major_label_text_font_size = '0pt'
  p.x_range.range_padding = 0.05
  p.xgrid.grid_line_color = None
  p.xaxis.axis_label = "Word Type"
  p.yaxis.axis_label = "Percent"

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Type of Dictionary", "@x"), ("Percent", "@percents")]
  p.add_tools(hover)
  
  script, div = components(p)
  
  return script, div

def get_letter_percents(words):
  first_letters = []
  for s in words:
    if s[0].isalpha():
      first_letters.append(s[0].upper())

  unique_letters, counts = np.unique(first_letters, return_counts=True)

  percents = []
  total = np.sum(counts)
  for c in counts:
    percents.append(c/total*100)

  return unique_letters, percents

def compare_first_letter(slang, non_slang):
  s_first, s_percents = get_letter_percents(slang)
  _, ns_percents = get_letter_percents(non_slang)

  s1, d1 = compare_first_deltas(s_first, s_percents, ns_percents)

  categories = ['Urban Dictionary', 'Standard English Dictionary']
  compare = s_first
  percents = []
  for i, p in enumerate(s_percents):
    percents.append(p)
    percents.append(ns_percents[i])

  x = [(c, category) for c in compare for category in categories]

  palette = ['#718dbf', '#e84d60']  # Blue for Definition, Red for Example
  palette=Category10[3]

  source = ColumnDataSource(data=dict(x=x, percents=percents, category=categories*26))

  p = figure(x_range=FactorRange(*x),
           height=400,
           title="Comparison of First Letters of Words")

  p.vbar(x='x', top='percents', width=0.8, source=source,
       fill_color=factor_cmap('x', palette=palette, factors=categories, start=1),legend_field='category')

  # Add aesthetics
  p.y_range.start = 0
  p.xaxis.major_label_text_font_size = '0pt'
  p.x_range.range_padding = 0.05
  p.xgrid.grid_line_color = None
  p.xaxis.axis_label = "First Letter"
  p.yaxis.axis_label = "Percent"

  p.legend.location = 'top_left'

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("First Letter", "@x"), ("Percent", "@percents")]
  p.add_tools(hover)
  
  script, div = components(p)
  
  return script, s1, div, d1

def compare_first_deltas(letters, s, ns):
  percent_delta = []
  for i, p in enumerate(s):
    percent_delta.append(abs(p - ns[i]))

  d = dict(zip(letters, percent_delta))

  sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

  letters = list(sorted_dict.keys())
  percent_delta = list(sorted_dict.values())

  p = figure(x_range=letters,\
             title="Difference in First Letter Percents of Words in Urban Dictionary and Standard English Dictionary",\
             x_axis_label='First Letter',\
             y_axis_label='Delta',\
             width=1000)

  # Create vertical bars
  p.vbar(x=letters, top=percent_delta, width=0.5, color=Category10[3][0])

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("First Letter", "@x"), ("Delta", "@top")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def phoenetics(slang, top=None):
  sound_counts = {}

  for s in slang:
    phones = pronouncing.phones_for_word(s)
    if not phones:
      continue
    for phone in phones[0].split():
      sound_counts[phone] = sound_counts.get(phone, 0) + 1

  sorted_dict = dict(sorted(sound_counts.items(), key=lambda item: item[1], reverse=True))

  sounds = list(sorted_dict.keys())
  counts = list(sorted_dict.values())
  title = "Frequency of Phones in Slang"

  if top is not None:
    sounds = sounds[0:top]
    counts = counts[0:top]
    title = f"Top {top} Phones in Slang"

  p = figure(x_range=sounds,\
             title=title,\
             x_axis_label='Phone',\
             y_axis_label='Frequency',\
             width=1000)

  # Create vertical bars
  p.vbar(x=sounds, top=counts, width=0.5, color=Category10[3][0])

  # Customize plot aesthetics
  p.xaxis.major_label_orientation = "vertical"
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Phone", "@x"), ("Frequency", "@top")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def get_phone_percents(words):
  p = []
  for s in words:
    phones = pronouncing.phones_for_word(s)
    if not phones:
      continue
    for phone in phones[0].split():
      p.append(phone)
  unique_phones, counts = np.unique(p, return_counts=True)

  percents = []
  total = np.sum(counts)
  for c in counts:
    percents.append(c/total*100)

  return unique_phones, percents

def compare_phones(slang, non_slang):
  s_first, s_percents = get_phone_percents(slang)
  _, ns_percents = get_phone_percents(non_slang)

  s1, d1 = compare_phone_deltas(s_first, s_percents, ns_percents)

  categories = ['Urban Dictionary', 'Standard English Dictionary']
  compare = s_first
  percents = []
  for i, p in enumerate(s_percents):
    percents.append(p)
    percents.append(ns_percents[i])

  x = [(c, category) for c in compare for category in categories]

  palette = ['#718dbf', '#e84d60']
  palette =  Category10[3]# Blue for Definition, Red for Example

  source = ColumnDataSource(data=dict(x=x, percents=percents, category=categories*len(s_first)))

  p = figure(x_range=FactorRange(*x),
           height=400,
           title="Comparison of Phones in Words",width=2000)

  p.vbar(x='x', top='percents', width=0.8, source=source,
       fill_color=factor_cmap('x', palette=palette, factors=categories, start=1),legend_field='category')

  # Add aesthetics
  p.y_range.start = 0
  p.xaxis.major_label_text_font_size = '0pt'
  p.x_range.range_padding = 0.05
  p.xgrid.grid_line_color = None
  p.xaxis.major_label_orientation = 0.785
  p.xaxis.axis_label = "Phone"
  p.yaxis.axis_label = "Percent"

  p.legend.location = 'top_right'

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Phone", "@x"), ("Percent", "@percents")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, s1, div, d1

def compare_phone_deltas(phones, s, ns):
  percent_delta = []
  for i, p in enumerate(s):
    percent_delta.append(abs(p - ns[i]))

  d = dict(zip(phones, percent_delta))

  sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

  phones = list(sorted_dict.keys())
  percent_delta = list(sorted_dict.values())

  p = figure(x_range=phones,\
             title="Difference in Phone Percents in Words of Urban Dictionary and Standard English Dictionary ",\
             x_axis_label='Phone',\
             y_axis_label='Delta',\
             width=1000)

  # Create vertical bars
  p.vbar(x=phones, top=percent_delta, width=0.5, color=Category10[3][0])

  # Customize plot aesthetics
  p.xaxis.major_label_orientation = "vertical"
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Phone", "@x"), ("Delta", "@top")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def num_words(slang):
  num_words = {}
  for s in slang:
    word_count = len(s.split())
    num_words[word_count] = num_words.get(word_count, 0) + 1

  sorted_dict = dict(sorted(num_words.items(), key=lambda item: item[1], reverse=True))

  counts = list(sorted_dict.keys())
  counts = [str(count) for count in counts]
  frequencies = list(sorted_dict.values())

  p = figure(x_range=counts,\
             title="Number of Words in Slang",\
             x_axis_label='Number of Words',\
             y_axis_label='Frequency')

  # Create vertical bars
  p.vbar(x=counts, top=frequencies, width=0.9, color=Category10[3][0])

  # Customize plot aesthetics
  p.xgrid.grid_line_color = None
  p.y_range.start = 0

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Number of Words", "@x"), ("Frequency", "@top")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div

def dash_percent(words):
  num_dashes = 0
  num_no_dashes = 0
  for s in words:
    if '-' in s:
      num_dashes += 1
    else:
      num_no_dashes += 1

  percent_dash = num_dashes/(num_dashes+ num_no_dashes)*100
  percent_no_dash = num_no_dashes/(num_dashes+ num_no_dashes)*100

  return percent_dash, percent_no_dash

def compare_dashes(slang, non_slang):
  percent_s_dash, percent_s_no_dash = dash_percent(slang)
  percent_ns_dash, percent_ns_no_dash = dash_percent(non_slang)

  categories = ['Urban Dictionary', 'Standard English Dictionary']
  percents = [percent_s_dash, percent_ns_dash, percent_s_no_dash, percent_ns_no_dash]
  compare = ["With Dashes", "Without Dashes"]

  x = [(c, category) for c in compare for category in categories]

  palette = ['#718dbf', '#e84d60']  # Blue for Definition, Red for Example
  palette = Category10[3]

  source = ColumnDataSource(data=dict(x=x, percents=percents, category=categories*2))

  p = figure(x_range=FactorRange(*x),
           height=400,
           title="Comparison of Words with Dashes vs No Dashes")

  # title = Comparison of Words with Dashes vs No Dashes in Urban Dictionary and Standard Dictionary"
  p.vbar(x='x', top='percents', width=0.8, source=source,
       fill_color=factor_cmap('x', palette=palette, factors=categories, start=1), legend_field='category')

  # Add aesthetics
  p.y_range.start = 0
  p.xaxis.major_label_orientation = "vertical"
  p.xaxis.major_label_text_font_size = '0pt'
  p.x_range.range_padding = 0.05
  p.xgrid.grid_line_color = None
  p.xaxis.axis_label = "Word Type"
  p.yaxis.axis_label = "Percent"

  p.legend.location = 'top_left'

  # Hover tool
  hover = HoverTool()
  hover.tooltips = [("Word Type", "@x"), ("Percent", "@percents")]
  p.add_tools(hover)

  script, div = components(p)
  
  return script, div