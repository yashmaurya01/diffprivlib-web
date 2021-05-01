import os
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template, url_for, abort, session
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import datasets

import matplotlib.pyplot as plt
from numpy.random import random
from diffprivlib import BudgetAccountant
from diffprivlib.tools import mean, var, nanmean, nanstd, nansum, nanvar, count_nonzero
from diffprivlib import tools as dp
import matplotlib.pyplot as plt

import string
import random

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'datasets'
app.config['UPLOAD_EXTENSIONS'] = ['.txt', '.csv', '.data', '.names']
app.secret_key = 'abc123'
app.config["CACHE_TYPE"] = "null"

cur_dir = os.getcwd()

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'

    return response


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        input_file, file_ext = os.path.splitext(filename)

        #Generating random alphanumeric string to avoid repetition of folder names
        letters = string.ascii_uppercase + string.digits
        folder = ''.join(random.choice(letters) for i in range(10))
        print(folder)
        session['folder'] = folder

        path = os.path.join(app.config['UPLOAD_PATH'],folder)
        
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            flash('Unsupported file type', 'error')
            return redirect(url_for('home'))

        try:
            os.mkdir(path)
        except OSError as error:
            flash(error, 'error')
            print(error)
            return redirect(url_for('home'))
 
        uploaded_file.save(os.path.join(path, filename))
        session['file'] = filename
        df = pd.read_csv(os.path.join(path,filename))
        # print(type(df))
        # session['df'] = df
        headers = list(df.columns)
        session['columns'] = headers

        df = df.head(5)
        dataset = df.to_html()
        session['dataset'] = dataset
        

        flash('File Uploaded!', 'info')
        
        session['path'] = path
        session['input_file'] = input_file
        session['output_file'] = input_file + '_parsed'

        session['filepath'] = os.path.join(session['path'], session['file'])

        return render_template('index.html', headers=session['columns'], data = dataset, preview = True, categorize=True)
    
    else:
        flash('No file specified', 'error')
        return redirect(url_for('upload_files'))

@app.route('/categories', methods=['POST'])
def categorize():
    session['epsilon'] = float(request.form['epsilon'])
    # session['delta'] = float(request.form['delta'])
    # session['slack'] = float(request.form['slack'])
    session['target'] = request.form['target']

    flash('Target Column assigned', 'info')

    df = pd.read_csv(session['filepath'])

    # acc = BudgetAccountant(session['epsilon'], session['delta'], session['slack'])
    arr = df[session['target']]

    plotname, hist_err = hist(df, session['target'])
    dp_count_nonzero = count_nonzero(arr, epsilon = session['epsilon'])
    dp_mean = nanmean(arr, epsilon = session['epsilon'])
    dp_std = nanstd(arr, epsilon = session['epsilon'])
    dp_sum = nansum(arr, epsilon = session['epsilon'])

    # df.drop(session['target'], inplace=True, axis=1)

    return render_template(
        'index.html',
        plotname = plotname,
        histogram=True, 
        hist_err= hist_err, 
        dp_mean = dp_mean,
        dp_count_nonzero = dp_count_nonzero,
        dp_std = dp_std,
        dp_sum = dp_sum
    )

# @app.route('/histogram', methods=['POST'])
# def histogram():
#     df = pd.read_csv(session['filepath'])

#     # acc = BudgetAccountant(session['epsilon'], session['delta'], session['slack'])
#     arr = df[session['target']]

#     plotname, hist_err = hist(df, session['target'])
#     dp_count_nonzero = count_nonzero(arr, epsilon = session['epsilon'])
#     dp_mean = nanmean(arr, epsilon = session['epsilon'])
#     dp_std = nanstd(arr, epsilon = session['epsilon'])
#     dp_sum = nansum(arr, epsilon = session['epsilon'])

#     df.drop(session['target'], inplace=True, axis=1)


#     # X_train, X_test, y_train, y_test = train_test_split(df, arr, test_size=0.2)
#     # print("Train examples: %d, Test examples: %d" % (X_train.shape[0], X_test.shape[0]))

#     # from diffprivlib.models import LinearRegression

#     # regr = LinearRegression()
#     # regr.fit(X_train, y_train)

#     # print("R2 score for epsilon=%.2f: %.2f" % (regr.epsilon, regr.score(X_test, y_test)))
    
    
#     return render_template(
#         'index.html',
#         plotname = plotname,
#         histogram=True, 
#         hist_err= hist_err, 
#         dp_mean = dp_mean,
#         dp_count_nonzero = dp_count_nonzero,
#         dp_std = dp_std,
#         dp_sum = dp_sum
#     )


# @app.route('/predict',methods=['POST'])
# def predict():
#     syn_data_files = []

#     # sample_tablegan(dataset_name, table_name, dataset_root_folder, output=None, sample_synthetic_rows=10000, tablegan_optional_parameters={}, preprocess_table=lambda x: x):

#     for i in range(1, 5):
#         # os.system(f"rm datasets/Hazards/LibertyMutualHazard.csv")
#         # os.system(f"cp datasets/Hazards/LibertyMutualHazard_train{i}.csv datasets/Hazards/LibertyMutualHazard.csv")
#         syn_data = sample.sample_tablegan(session['folder'], session['input_file'], "./datasets", sample_synthetic_rows=1000)
#         syn_data_files.append(syn_data)
#     # sample.sample_tablegan("Hazards", "LibertyMutualHazard", "./datasets", output=f"datasets/Hazards/LibertyMutualHazard_train_output{i}.csv", sample_synthetic_rows=41600, preprocess_table=preprocess_hazards)

#     new_data = pd.concat(syn_data_files)
#     new_data.to_csv('new_data.csv')
#     flash('Dataset Generation Complete!', 'info')
#     return render_template('index.html', data = session['dataset'], gendata=new_data.to_html(), generated=True, preview=True)

# Download API
# @app.route("/download", methods = ['GET'])
# def download_file():
#     output = 'new_data.csv'
#     file_path = os.path.join(cur_dir,output)
#     flash('Dataset Downloading...', 'info')
#     return send_file(file_path, as_attachment=True)


def hist(dataset, target):
    # print(f'Remaining Budget = {acc.remaining()}')

    dataset = dataset.dropna()

    hist, bins = np.histogram(dataset[target].to_numpy())
    hist = hist / hist.sum()
    
    dp_hist, dp_bins = dp.histogram(dataset[target].to_numpy())
    dp_hist = dp_hist / dp_hist.sum()

    print("Total histogram error: %f" % np.abs(hist - dp_hist).sum())
    hist_err = np.abs(hist - dp_hist).sum()

    plt.bar(dp_bins[:-1], dp_hist, width=(dp_bins[1] - dp_bins[0]) * 0.9, color= '#e57184')
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.savefig('static/'+ session['folder'] +'new_plot.png',transparent=True)

    plt.clf()
    plt.cla()
    plt.close()
    return session['folder'] +'new_plot.png', hist_err



if __name__ == "__main__":
    app.run(debug=True)



