from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
clustrer = joblib.load('clustering_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/one')
def one():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    formated_data = []
    
    for row in data['data']:
        correct_data_type = []
        for index in row:
            try:
                correct_data_type.append(float(index))
            except:
                correct_data_type.append(index.replace('"',''))
        formated_data.append(correct_data_type)

    try:
        df = pd.DataFrame(formated_data,columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
            'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
            'cons.conf.idx', 'euribor3m', 'nr.employed'])
        df.drop(['default','duration'],axis=1,inplace=True)
    except:
        df = pd.DataFrame(formated_data,columns=["age", "job", "marital", "education", "housing",
        "loan", "contact", "month", "day_of_week", "campaign",
        "pdays", "previous", "poutcome"])
        nan_df = pd.DataFrame([[-1.8,92.893,-46.2,1.313,5099.1]],columns=['emp.var.rate', 'cons.price.idx',
            'cons.conf.idx', 'euribor3m', 'nr.employed'])
        df = pd.concat([df,nan_df],axis=1)

    df = df.replace('unknown',np.nan)

    def get_season(month):
        if month in ['mar', 'apr', 'may']:
            return 'spring'
        elif month in ['jun', 'jul', 'aug']:
            return 'summer'
        elif month in ['sep', 'oct', 'nov']:
            return 'fall'
        else:
            return 'winter'
    
    df['season'] = df['month'].apply(get_season)
    df['has_any_loan'] = df.apply(lambda x: 1 if x['housing'] == 'yes' or x['loan'] == 'yes' else 0, axis=1)
    df['contacted_before'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1)
    df['age_group'] = pd.cut(df['age'],5,labels=['young','adult','middle_aged','senior','elderly'])

    df["economic_index"] = (
        df["emp.var.rate"] +
        df["cons.price.idx"] +
        df["cons.conf.idx"] +
        df["euribor3m"] +
        df["nr.employed"]
    )
    def campaign_length(x):
        if x == 1:
            return "short"
        elif 2 <= x <= 4:
            return "medium"
        else:
            return "long"

    df["campaign_category"] = df["campaign"].apply(campaign_length)
    df["interaction_intensity"] = df.apply(
        lambda row: row["campaign"] / (row["previous"] + 1), axis=1
    )

    clusters_feat = clustrer.predict(df)

    df_new = np.c_[df,clusters_feat]

    df_new_df = pd.DataFrame(df_new,index=df.index,columns=list(df.columns) + ['clusters'])

    predictions = model.predict(df_new_df)

    return jsonify({"predictions": ['yes' if i == 1 else 'no' for i in predictions]})

if __name__ == "__main__":
    app.run(debug=True)