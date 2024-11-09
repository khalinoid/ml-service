from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np

app = Flask(__name__)

# Sample training data
training_data = [
    "This is amazing!", 
    "I love this product",
    "Great service",
    "Terrible experience",
    "Very disappointed",
    "Waste of money"
]
training_labels = np.array([1, 1, 1, 0, 0, 0])  # 1 for positive, 0 for negative

# Train the model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_data)
classifier = MultinomialNB()
classifier.fit(X, training_labels)

# Save the model and vectorizer
joblib.dump(classifier, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        # Transform the input text
        text_vectorized = vectorizer.transform([text])
        
        # Make prediction
        prediction = classifier.predict(text_vectorized)[0]
        probability = classifier.predict_proba(text_vectorized)[0]
        
        return jsonify({
            'text': text,
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(max(probability))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Changed port to 5000