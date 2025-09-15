from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

app = Flask(__name__)

@app.route('/')
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['review']
    if not input_text.strip():
        return render_template('analyze.html', prediction="Please enter text.")

    vectorized_input = vectorizer.transform([input_text])
    raw_pred = model.predict(vectorized_input)[0]

    # Sentiment visual aids
    if raw_pred == "positive":
        emoji = "😍"  # heart eyes
        pred_class = "positive"
    elif raw_pred == "negative":
        emoji = "😠"  # angry face
        pred_class = "negative"
    else:
        emoji = "😐"  # neutral face
        pred_class = "neutral"

    # ✅ This must be indented inside the function
    return render_template(
        'analyze.html',
        prediction=f"Sentiment: {raw_pred.capitalize()}",
        prediction_class=pred_class,
        emoji=emoji
    )


if __name__ == '__main__':
    app.run(debug=True)
