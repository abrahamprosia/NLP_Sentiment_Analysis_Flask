import pickle
from flask import Flask, render_template, request
import nltk
nltk.download('wordnet')




app = Flask(__name__)

# Loading models and vectorizer
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
lemmatizer = pickle.load(open('lemmatizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('input_text', '').strip()

    if not input_text:
        return render_template('index.html', prediction="Input text cannot be empty.")

    input_text = input_text.split()
    input_text = [lemmatizer.lemmatize(word) for word in input_text]
    input_text = ' '.join(input_text)

    input_text = vectorizer.transform([input_text])

    predicted = nb_model.predict(input_text)[0]  # Ensure you're accessing the first prediction

    # Preparing the result message
    if predicted == 0:
        result = "The sentiment is Positive :)"
    elif predicted == 1:
        result = "The sentiment is Negative :("
    else:
        result = "The sentiment is just Neutral :|"


    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
