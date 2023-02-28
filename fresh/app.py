import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from flask import Flask, request
from flask_restful import Api, Resource, reqparse

# Download the required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define the summarization function
def summarize(text, n=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        filtered_sentences.append(' '.join(words))
    # Calculate the frequency distribution of words
    words = word_tokenize(' '.join(filtered_sentences))
    fdist = FreqDist(words)
    # Sort the sentences by their frequency-weighted scores
    scores = [(i, sum(fdist[word] for word in word_tokenize(sentence))) for i, sentence in enumerate(sentences)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # Return the top n sentences as the summary
    summary = [sentences[score[0]] for score in scores[:n]]
    return ' '.join(summary)

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('text', type=str, required=True, help='Text to summarize')

class Summarizer(Resource):
    def post(self):
        args = parser.parse_args()
        text = args['text']
        # Apply the summarization model to the input text
        summary = summarize(text)
        return {'summary': summary}

api.add_resource(Summarizer, '/summarize')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
