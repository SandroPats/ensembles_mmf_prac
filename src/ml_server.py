import os
import pickle
import datetime

import pandas as pd
import numpy as np

import custom_forms as cf
from wrapper_classes import Dataset, Model
from exceptions import DatasetError, TargetError
from plotting_funcs import plot_and_save_validation
import plotly
import plotly.subplots
import plotly.graph_objects as go
from shapely.geometry.polygon import Point
from shapely.geometry.polygon import Polygon

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, session, send_from_directory
from flask import render_template, redirect


app = Flask(__name__, template_folder='templates')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)


model_dict = {}
data_dict = {}


@app.route('/')
@app.route('/index')
def get_index():
    return render_template('index.html')


@app.route('/new_model', methods=['POST', 'GET'])
def create_model():
    model_selection = cf.ModelSelectionForm(meta={'csrf': False})

    if model_selection.validate_on_submit():
        model = model_selection.model.data
        name = model_selection.name.data
        return redirect(url_for('set_model', model=model, name=name))

    return render_template('model_selection.html', form=model_selection)


@app.route('/new_model/set_model/<model>_<name>', methods=['POST', 'GET'])
def set_model(model, name):
    model_params = cf.ModelParamsForm(meta={'csrf': False})
    
    if model_params.validate_on_submit():
        if model == 'RF':
            del model_params.learning_rate
        del model_params.submit
        model_dict[name] = Model(model, name, model_params)
        return redirect(url_for('get_models'))

    return render_template('model_settings.html',
                           form=model_params,
                           model=model,
                           name=name)


@app.route('/saved_datasets/data/<invalid_col>', methods=['POST', 'GET'])
def create_data(invalid_col):
    data_upload = cf.FileUploadForm(meta={'csrf': False})

    if data_upload.validate_on_submit():
        data = pd.read_csv(data_upload.data_file.data)
        target_col = data_upload.target_col.data
        name = data_upload.data_name.data
        cat_strat = data_upload.cat_nan_processing.data
        num_strat = data_upload.num_nan_processing.data
        try:
            dataset = Dataset(data_upload.data_name.data, data,
                              target_col, cat_strat, num_strat)
        except TargetError as te:
            app.logger.info('Exception: {0}'.format(te))
            return redirect(url_for('create_data', invalid_col=True))
        data_dict[name] = dataset
        return redirect(url_for('get_data'))
        
    return render_template('data_load.html',
                           form=data_upload,
                           invalid_target_col=invalid_col)


@app.route('/saved_models')
def get_models():
    not_empty = False if not model_dict else True
    return render_template('saved_models.html',
                           models=model_dict,
                           not_empty=not_empty)


@app.route('/saved_datasets')
def get_data():
    not_empty = False if not data_dict else True
    return render_template('saved_datasets.html',
                           datasets=data_dict,
                           not_empty=not_empty)


@app.route('/saved_datasets/data_info_<name>')
def get_data_info(name):
    dataset = data_dict[name]
    
    return render_template('data_info.html',
                           dataset=dataset)


@app.route('/saved_models/model_page_<name>/val_score')
def get_val_score(name):
    val_df = model_dict[name].get_val_score()

    return render_template('val_score.html', name=name, val_score=val_df)


@app.route('/saved_models/model_page_<name>', methods=['POST', 'GET'])
def get_model_page(name):
    model = model_dict[name]
    train_form = cf.TrainForm(meta={'csrf': False})
    test_form = cf.TestForm(meta={'csrf': False})
    not_empty = False if not data_dict else True
    data_with_target = []

    for data_name, data in data_dict.items():
        if data.has_target():
            data_with_target.append(data_name)

    train_form.train.choices = data_with_target
    train_form.val.choices = ['Отсутствует'] + data_with_target
    test_form.test.choices = list(data_dict.keys())
    model = model_dict[name]

    if 'png_path' in session:
        png_path = session['png_path']
    else:
        png_path = None

    if train_form.validate_on_submit():
        train_name = train_form.train.data
        val_name = train_form.val.data
        
        if val_name == 'Отсутствует':
            val_name = train_name

        train = data_dict[train_name]
        val = data_dict[val_name]

        try:
            val_score = model.fit(train, val)
        except DatasetError as de:
            app.logger.info('Exception: {0}.'.format(te))
            return redirect(url_for('get_model_page', name=name))

        path = os.path.join(os.getcwd(), 'static/tmp/')
        if not os.path.exists(path):
            os.mkdir(path)
        png_path = os.path.join(path, '{0}.png'.format(name))
        plot_and_save_validation(val_score, png_path)
        png_path = '../static/tmp/{0}.png'.format(name)
        session['png_path'] = png_path

        return redirect(url_for('get_model_page', name=name))
    
    if test_form.validate_on_submit():
        test_name = test_form.test.data
        test = data_dict[test_name]

        try:
            pred_df = model.predict(test)
        except DatasetError as de:
            app.logger.info('Exception: {0}.'.format(te))
            return redirect(url_for('get_model_page', name=name))
        
        pred_name = test_name + '_prediction.csv'
        path = os.path.join(os.getcwd(), 'static/tmp/')
        if not os.path.exists(path):
            os.mkdir(path)
        pred_df.to_csv(os.path.join(path, pred_name))
        return send_from_directory(path, pred_name, as_attachment=True)
    
    return render_template('model_page.html',
                           model=model,
                           train_form=train_form,
                           test_form=test_form,
                           is_empty=not data_with_target,
                           png_path=png_path)
