import json

def get_data():
    with open('urban_dict_data_cleaned_emo.json', 'r') as file:
        urban_dict_data = json.load(file)
    return urban_dict_data