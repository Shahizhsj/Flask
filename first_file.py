from flask import Flask,request
from finvizfinance.screener.overview import Overview
from flask_jsonpify import jsonpify
import yfinance as yf
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)

@app.route("/filter")
def index():
    foverview = Overview()
    filters_dict = {'EPS growthqtr over qtr':'Positive (>0%)','P/E':'Over 10'}
    query_1=request.args.get('name_1')
    query_2=request.args.get('name_2')
    query={'EPS growthqtr over qtr':str(query_1),'P/E':str(query_2)}
    foverview.set_filter(filters_dict=query)
    df = foverview.screener_view()
    df=df.values.tolist()
    json=jsonpify(df)
    return json
@app.route('/predict')
def predict():
    symbol=request.args.get('sys')
    sp500 = yf.Ticker(str(symbol))
    sp500 = sp500.history(period="max")
    sp500.index = pd.to_datetime(sp500.index)
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    horizons = [2,5,60,250,1000]
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        predictors+= [ratio_column, trend_column]

    sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(sp500[predictors],sp500['Target'])
    res=model.predict(sp500.tail(1)[predictors])
    if res==0:
        return jsonpify(prediction='Price may fall down')
    else:
        return jsonpify(prediction='Price may increase')
# Run the Flask app.
app.run(debug=False,host='0.0.0.0')
