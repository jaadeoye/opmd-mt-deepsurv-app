from flask import Flask, render_template, url_for, request
import pandas as pd 
import numpy as np 
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
import torch
import torchtuples as tt
import celery
from pycox.models import CoxPH
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def analyze():
    if request.method == 'POST':
        P1 = request.form['P1']
        P2 = request.form['P2']
        P3 = request.form['P3']
        P4 = request.form['P4']
        P5 = request.form['P5']
        P6 = request.form['P6']
        P7 = request.form['P7']
        P8 = request.form['P8']
        P9 = request.form['P9']
        P10 = request.form['P10']
        P11 = request.form['P11']
        P12 = request.form['P12']
        P13 = request.form['P13']
        P14 = request.form['P14']
        P15 = request.form['P15']
        P16 = request.form['P16']
        P17 = request.form['P17']
        P18 = request.form['P18']
        P19 = request.form['P19']
        P20 = request.form['P20']
        P21 = request.form['P21']
        P22 = request.form['P22']
        P23 = request.form['P23']
        P24 = request.form['P24']
        P25 = request.form['P25']
        P26 = request.form['P26']
        model_choice = request.form['model_choice']

        sample_data = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23,
                       P24, P25, P26]
        clean_data = [float(i) for i in sample_data]
        x1 = np.array(clean_data)
        x2 = torch.from_numpy(x1)
        x3 = x2.type(torch.float32)
        ex1 = x3.unsqueeze(0)

        if model_choice == 'deepsurv':
            np.random.seed(1234)
            _ = torch.manual_seed(123)
            df_train = pd.read_csv('static/MTP_surv_.csv')
            df_val = df_train.sample(frac=0.2)
            cols_leave = ['P1', 'P2', 'P3', 'P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14',
                             'P15','P16','P17','P18','P19','P20','P21','P22','P23','P24',
                          'P25','P26',]
            leave = [(col, None) for col in cols_leave]
            x_mapper = DataFrameMapper(leave)
            x_train = x_mapper.fit_transform(df_train).astype('float32')
            x_val = x_mapper.transform(df_val).astype('float32')
            get_target = lambda df: (df['time'].values, df['status'].values)
            y_train = get_target(df_train)
            y_val = get_target(df_val)
            val = x_val, y_val
            in_features = x_train.shape[1]
            num_nodes = [64, 64, 64]
            out_features = 1
            batch_norm = True
            dropout = 0.4
            output_bias = False
            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                              batch_norm,dropout, output_bias=output_bias)
            model = CoxPH(net, tt.optim.Adam)
            batch_size = 64
            lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
            lrfinder.get_best_lr()
            model.optimizer.set_lr(0.01)
            epochs = 512
            callbacks = [tt.callbacks.EarlyStopping()]
            verbose = True
            log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                            val_data=val, val_batch_size=batch_size)
            model.partial_log_likelihood(*val).mean()
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(ex1)
            plt.plot(surv, color='red', linewidth=2)
            plt.xlabel("Duration in months")
            plt.ylabel("MT-free probability")
            plt.savefig('static/dps.jpg',  bbox_inches='tight')

        return render_template('predict.html', result_prediction = 'Prediction plot', url = 'static/dps.jpg', model_selected=model_choice)
       

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=9030)
