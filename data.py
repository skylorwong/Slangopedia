import json
import pandas as pd
import fasttext
import numpy as np
#from sklearn.decomposition import PCA

def get_data():
    urban_dict_data = {}
    data_str = 'urbandict2.'
    jsons = [f'{data_str}0',
             f'{data_str}1',
             f'{data_str}2',
             f'{data_str}3',
             f'{data_str}4',
             f'{data_str}5']
    for data in jsons:
        with open(f'urbandict2/{data}.json', 'r') as file:
            urban_dict_data = urban_dict_data | json.load(file)
    return urban_dict_data

def get_date_df_all(urban_dict):
  data = []
  for word, info in urban_dict.items():
    for e in info['top_5_entries']:
      entry = {}
      entry['word'] = word
      entry['date'] = e['date'][0:10]
      data.append(entry)

  date_df = pd.DataFrame(data)

  date_df['date'] = pd.to_datetime(date_df['date'])
  date_df['year'] = date_df['date'].dt.year
  date_df['month_datetime'] = date_df['date'].dt.to_period('M').dt.to_timestamp()
  date_df['year_datetime'] = pd.to_datetime(date_df['year'], format='%Y')

  return date_df

def get_date2(urban_dict):
    data = []
    for word, info in urban_dict.items():
        for t in info['top_5_entries']:
            entry = {}
            entry['word'] = word
            entry['d_sentiment'] = t['definition_sentiment_label']
            entry['d_emotion'] = t['definition_emotion_label']
            entry['e_sentiment'] = t['example_sentiment_label']
            entry['e_emotion'] = t['example_emotion_label']
            entry['date'] = t['date'][0:10]
            data.append(entry)

    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    exclude = ['trust', 'pessism', 'love']
    df_removed = df[df['d_emotion'] != 'trust']
    df_removed = df_removed[df_removed['d_emotion'] != 'pessism']
    df_removed = df_removed[df_removed['d_emotion'] != 'love']
    df_removed = df_removed[df_removed['e_emotion'] != 'pessism']
    df_removed = df_removed[df_removed['e_emotion'] != 'trust']
    df_removed = df_removed[df_removed['e_emotion'] != 'love']

    return df_removed


def get_non_slang():
   df = pd.read_csv('englishdict.csv')
   
   columns_as_lists = {col: df[col].tolist() for col in df.columns}
   english_words = columns_as_lists['word']
   non_slang = []
   for i, w in enumerate(english_words):
        if not isinstance(w, str):
           w = str(w)
        non_slang.append(w)
   return non_slang

def get_nearest(word):
    with open('nearest.json', 'r') as f:
       data = json.load(f)
    return data[word]
   


