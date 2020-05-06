import pandas as pd
import numpy as np
from flask import Flask,json,request,jsonify,render_template,redirect
import requests
import datetime
import pickle
from xgboost import Booster
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from keras.models import model_from_json
# from keras_retinanet.models import load_model
# from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

win_loss_draw = Booster()
win_loss_draw.load_model('./static/models/win_loss_draw.model')
win_type = Booster()
win_type.load_model('./static/models/win_type.model')
print('model loaded')

# def loadmodel(modelname,weightname):
#     json_file = open(modelname, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights(weightname)
#     print("Loaded model from disk")
#     return loaded_model

@app.route('/getImage')
def getImage():
    name = request.args.get('name')
    print(name,'name')
    df = pd.read_csv('fighters_images.csv')
    print(df)
    img = df.loc[df['name']==name]['image']
    print(img)
    if not img.empty:
        img=img.item()
    else:
        img ='https://www.ufc.com/themes/custom/ufc/assets/img/no-profile-image.png'

    return jsonify({'image':img})

@app.route('/')
def start():
    filename='./static/csv/fighter_names.csv'
    df = pd.read_csv(filename)
    names = df['Name']
    return render_template('index.html',names=names)

def encode_winning(winning):
    if winning == 'Decision - Unanimous':
        return 1
    elif winning == 'KO/TKO':
        return 2
    elif winning == 'Submission':
        return 3
    elif winning == 'Decision - Split':
        return 4
    elif winning == 'TKO - Doctor\'s Stoppage':
        return 5
    elif winning == 'Decision - Majority':
        return 6
    elif winning == 'Overturned':
        return 7
    elif winning == 'DQ':
        return 8
    elif winning == 'Could Not Continue':
        return 9
    elif winning == 'Other':
        return 10

def encode_stance(stance):
    if stance == 'Orthodox':
        return 1
    elif stance == 'Southpaw':
        return 2
    elif stance == 'Switch':
        return 3
    elif stance == 'Open Stance':
        return 4
    elif stance == 'Sideways':
        return 5

def encode_names(name,fighters_Data):
    return int(fighters_Data[ fighters_Data['Name'] == name ].loc[:,'Encode_IDs'])

def preprocess(data,fighters_Data,isTest=False):
    data['R_fighter'] = data['R_fighter'].apply(lambda x: encode_names(x,fighters_Data))
    data['B_fighter'] = data['B_fighter'].apply(lambda x: encode_names(x,fighters_Data))
    if isTest == False:
        data['Winner'] = data['Winner'].replace('Red', 1)
        data['Winner'] = data['Winner'].replace('Blue', 2)
        data['Winner'] = data['Winner'].replace('Draw', 3)
    data['B_Stance'] = data['B_Stance'].apply(lambda x: encode_stance(x))
    data['R_Stance'] = data['R_Stance'].apply(lambda x: encode_stance(x))
    if isTest == False:
        data['win_by'] = data['win_by'].apply(lambda x: encode_winning(x))
    # data.fillna(-999,inplace=True)
    data['title_bout'] = data['title_bout'].apply(lambda x: 1 if x == True else 2)

    return data


@app.route("/predictions", methods=['GET', 'POST'])
def predictions():
    d1 = pd.read_csv('./static/csv/data.csv')
    d2 = pd.read_csv('./static/csv/raw_total_fight_data.csv', sep=';')
    fighters_Data = pd.read_csv('./static/csv/fighters_Data.csv')
    data = d1.copy()
    data['win_by'] = d2['win_by'].copy()
    cols_R = []
    cols_B = []
    only_cols = []
    for col in data.columns:
        if str(col).startswith('R_'):
            cols_R.append(col)
    for col in data.columns:
        if str(col).startswith('B_'):
            cols_B.append(col)
    cols_R.remove('R_fighter')
    cols_B.remove('B_fighter')
    for val in cols_R:
        val = val.replace('R_', '')
        only_cols.append(val)
    to_drop = ['Referee', 'location', 'weight_class', 'date']
    data.drop(to_drop, axis=1, inplace=True)
    numeric_data = data.copy()
    B_fighter =request.args.get("blue")
    R_fighter  = request.args.get("red")
    rounds = int(request.args.get("rounds"))
    title_bout = request.args.get("fight")
    if rounds=='True':
        no_of_rounds = True
    else:
        no_of_rounds = False
    print(title_bout,B_fighter,R_fighter,no_of_rounds,'sadsdasd')
    test = pd.DataFrame(columns=list(numeric_data.drop(['Winner', 'win_by'], axis=1).columns))
    test.at[0, 'R_fighter'] = R_fighter
    test.at[0, 'B_fighter'] = B_fighter
    test.at[0, 'title_bout'] = title_bout
    test.at[0, 'no_of_rounds'] = no_of_rounds
    test.at[0, cols_B] = fighters_Data[fighters_Data['Name'] == B_fighter].head(1).loc[:, only_cols].values
    test.at[0, cols_R] = fighters_Data[fighters_Data['Name'] == R_fighter].head(1).loc[:, only_cols].values
    test_numeric = preprocess(test,fighters_Data,isTest=True).values
    print(test_numeric,'test_numeric')
    test_numeric = xgb.DMatrix(test_numeric)
    results = win_loss_draw.predict(test_numeric)
    print(results,'results')
    types = win_type.predict(test_numeric)
    print(types,'types')
    res = {'R_prob':np.round(results[0,0]*100,2),
           'B_prob':np.round(results[0,1]*100,2),
           'draw': np.round(results[0,2]*100,2)
           }
    w_type = {
        'decision_unanimous':np.round(types[0,0]*100,2),
        'ko':np.round(types[0,1]*100,2),
        'sub':np.round(types[0,2]*100,2),
        'dec_split':np.round(types[0,3]*100,2),
        'doc': np.round(types[0,4]*100,2),
        'dec_maj':np.round(types[0,5]*100,2),
        'overt': np.round(types[0, 6] * 100, 2),
        'dq': np.round(types[0, 7] * 100, 2),
        'could_not_cont': np.round(types[0, 8] * 100, 2),
        'other': np.round(types[0, 9] * 100, 2),
    }
    return jsonify({'result':res, 'w_type':w_type})

if __name__ == '__main__':
    app.run('0.0.0.0', 8085 ,debug=True)
