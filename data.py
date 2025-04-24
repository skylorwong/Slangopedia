import json
import pandas as pd
import fasttext
import numpy as np
#from sklearn.decomposition import PCA

def get_data():
    with open('urban_dict_data_cleaned_emo.json', 'r') as file:
        urban_dict_data = json.load(file)
    return urban_dict_data

def get_date_df(urban_dict):
    data = []
    for word, info in urban_dict.items():
        entry = {}
        entry['word'] = word
        entry['date'] = info['top_5_entries'][0]['date'][0:10]
        data.append(entry)

    date_df = pd.DataFrame(data)

    date_df['date'] = pd.to_datetime(date_df['date'])
    date_df['year'] = date_df['date'].dt.year
    date_df['month_datetime'] = date_df['date'].dt.to_period('M').dt.to_timestamp()
    date_df['year_datetime'] = pd.to_datetime(date_df['year'], format='%Y')

    return date_df

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

def get_trends_df(urban_dict):
    data = []
    for word, info in urban_dict.items():
        entry = {}
        entry['word'] = word
        entry['d_sentiment'] = info['top_5_entries'][0]['definition_sentiment_label']
        entry['d_emotion'] = info['top_5_entries'][0]['definition_emotion_label']
        entry['e_sentiment'] = info['top_5_entries'][0]['example_sentiment_label']
        entry['e_emotion'] = info['top_5_entries'][0]['example_emotion_label']
        entry['date'] = info['top_5_entries'][0]['date'][0:10]
        data.append(entry)
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
   
    return df

def run_pca(slang):
  model = fasttext.load_model("urban_slang_ft.bin")
  word_vectors = np.array([model.get_word_vector(word) for word in slang])

  pca = PCA(n_components=2)
  pca_result = pca.fit_transform(word_vectors)

  return pca_result