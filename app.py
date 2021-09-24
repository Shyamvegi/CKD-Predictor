from flask import Flask, render_template, url_for, request
from sklearn import tree
import numpy as np
from statistics import mode
app = Flask(__name__, static_folder='./static/')


data = open('./static/data.txt', 'r')

features = []
labels = []

for row in data:
    row = row.split(',')

    if row[-1] == 'ckd\n':
        labels.append(1)
    else:
        labels.append(0)
    row.pop()
    row.pop(0)

    con = {
        'yes': 1,
        'no': 0,
        'notpresent': 0,
        'present': 1,
        'normal': 0,
        'abnormal': 1,
        'good': 1,
        'bad': 0,
        '': 0
    }

    feat = []

    for value in row:
        try:
            temp = float(value)
            feat.append(temp)
        except:
            feat.append(con.get(value))

    temp = []
    for i in feat:
        if i is None:
            temp.append(mode(feat))
        else:
            temp.append(i)
    feat = temp
    features.append(feat)

clf = tree.DecisionTreeClassifier()

clf.fit(features, labels)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/model', methods=['POST', 'GET'])
def runModel():
    data = request.get_json()['fields']
    result = clf.predict([data])
    return str(result).split('[')[1].split(']')[0]


if __name__ == "__main__":
    app.run(debug=True)
