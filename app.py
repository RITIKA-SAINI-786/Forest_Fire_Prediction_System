from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            int_features = [int(x) for x in request.form.values()]
            final = np.array(int_features).reshape(1, -1)
            print("Form values:", int_features)
            print("Final array:", final)
            prediction = model.predict_proba(final)
            output = '{0:.2f}'.format(prediction[0][1])
            if float(output) > 0.5:
                return render_template('forest.html',
                                       pred='Your Forest is in Danger.\nProbability of fire occurring is {}'.format(output))
            else:
                return render_template('forest.html',
                                       pred='Your Forest is safe.\nProbability of fire occurring is {}'.format(output))
        except Exception as e:
            print("Error:", e)
            return render_template('error.html', message='Error: Failed to make prediction')
    else:
        return render_template('error.html', message='Error: Invalid request method')

if __name__ == '__main__':
    app.run(debug=True)
