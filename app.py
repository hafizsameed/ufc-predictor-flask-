import pandas as pd
import numpy as np
from flask import Flask,json,request,jsonify,render_template,redirect
import requests
import datetime
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


def loadmodel(modelname,weightname):
    json_file = open(modelname, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightname)
    print("Loaded model from disk")
    return loaded_model




def daily_price_historical(symbol, comparison_symbol, all_data=True, limit=1, aggregate=1, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    if all_data:
        url += '&allData=true'
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df

@app.route('/')
def start():
    return render_template('index.html')


@app.route("/data", methods=['GET', 'POST'])
def data():
    name=request.args.get('modelname')
    weightname=request.args.get('weightname')
    symbol=request.args.get('symbol') 
    bitcoin = daily_price_historical(symbol,"USD")
    bitcoin_social = bitcoin[bitcoin.timestamp.dt.year >= 2020].reset_index().drop(['index'],axis=1)
    X = bitcoin_social.drop(['timestamp','close'],axis=1)
    X_transformed = StandardScaler().fit_transform(X)
    X_transformed = X_transformed.reshape((X_transformed.shape[0],X_transformed.shape[1],1))
    _y = bitcoin_social['close']
    model = loadmodel(name,weightname)
    model.compile(optimizer='Adam',loss='mae')
    model.fit(X_transformed,_y,verbose=0,epochs=100)
    y_hat = pd.DataFrame(model.predict(X_transformed))
    y_hat['pred'] = y_hat[0]
    y_hat['close']= bitcoin_social['close']
    y_hat['time']=bitcoin_social['time']
    y_hat.drop(columns=[0])
    actual = y_hat['close'].tolist()
    time = y_hat['time'].tolist()
    pred = y_hat['pred'].tolist()
    return jsonify({"y_hat":pred,"time":time,"actual":actual})

def getdata(name, symbol):
    df = daily_price_historical(symbol,"USD")
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by ='timestamp',inplace=True)
    close = df['close'].tolist()
    timestamp = df['timestamp'].tolist()
    return close,timestamp

@app.route("/predictions", methods=['GET', 'POST'])
def predictions():
    filename=request.args.get("filename")
    name = request.args.get("name")
    symbol = request.args.get("symbol")
    df = pd.read_csv(filename)
    df=df.dropna()
    prev_close, prev_time = getdata(name,symbol)
    # print(prev_time[len(prev_time)-1],"current last date")
    last_date=prev_time[len(prev_time)-1].strftime("%Y-%m-%d %H:%M:%S")
    # print(last_date,'last date')
    future_time = df['timestamp'].tolist()
    date_05=last_date[0:12]+'5'+last_date[13:len(last_date)]
    date_00=last_date[0:12]+'0'+last_date[13:len(last_date)]
    pred = df['pred'].tolist()
    conv_time=[]
    conv_close=[]
    flag=False
    for i in range(0,len(future_time)-1):
        if flag:
            conv_time.append(future_time[i])
            conv_close.append(pred[i])
        date=future_time[i]
        # print(date,"date")
        if date==date_00 or date==date_05:
            # print('true',future_date[i])
            flag=True
    # print(len(conv_time),"length")
    return jsonify({"prev_time":prev_time,"prev_close":prev_close,"future_time":conv_time,"future_close":conv_close})