import flask
import data
import visualizations as vis
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import HoverTool
import numpy as np

app = flask.Flask(__name__, template_folder='.')
urban_dict_data = data.get_data()
date_df = data.get_date_df(urban_dict_data)
date_df_all = data.get_date_df_all(urban_dict_data)
trends_df = data.get_trends_df(urban_dict_data)
slang = []
for key in urban_dict_data:
  slang.append(key)
#pca = data.run_pca(slang)
pca = None

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    html_code = flask.render_template('index.html')
    response = flask.make_response(html_code)
    return response

@app.route('/language', methods=['GET'])
def language():
    scripts, divs = vis.get_language_graphs(slang)
    html_code = flask.render_template('language.html', scripts=scripts, divs=divs)
    response = flask.make_response(html_code)
    return response

@app.route('/explore', methods=['GET'])
def explore():
    scripts, divs, p_graph = vis.get_explore_graphs(urban_dict_data, slang, date_df, date_df_all, pca)
    html_code = flask.render_template('explore.html', scripts=scripts, divs=divs, p_graph=p_graph)
    response = flask.make_response(html_code)
    return response

@app.route('/deeper', methods=['GET'])
def deeper():
    scripts, divs = vis.get_deeper_graphs(urban_dict_data)
    html_code = flask.render_template('deeper.html', scripts=scripts, divs=divs)
    response = flask.make_response(html_code)
    return response

@app.route('/trends', methods=['GET'])
def trends():
    scripts, divs, graphs = vis.get_trends_graphs(trends_df)
    html_code = flask.render_template('trends.html', scripts=scripts, divs=divs, graphs=graphs)
    response = flask.make_response(html_code)
    return response

@app.route('/search', methods=['GET'])
def search():
    html_code = flask.render_template('search.html')
    response = flask.make_response(html_code)
    return response