# Importing necessary libraries
from flask import Flask, render_template, request, jsonify,send_from_directory
import nltk
import numpy as np
import pickle
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import time
from sklearn.metrics.pairwise import cosine_similarity
# Create Flask app
app = Flask(__name__)
CORS(app)
# Load the saved model and vectorizer
model = load_model('chatbot_model.h5')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
 
# Define the corpus and responses
corpus = [
    'Hi there!',
    'How are you?',
    'What is your name?',
    'Can you help me?',
    'I want your contact details.',
    'Where is your office located?',
    'How can I reach your HR department?',
    'Who is your founder and CEO?',
    'What is your founder name?',
    'When was the organization founded?',
    'How old are you?',
    'What is the weather like today?',
    'Goodbye!',
    'Bye',
]
 
responses = [
    'Hello! Happy to welcome you.',
    'I am fine, thank you!',
    'My name is Thiksebot.',
    'Yes, I can help you. What do you need?',
    'Contact number: +91 4146 358 357.',
    '746/D 2nd floor neruji road,villupuram 605602.',
    'For career-related tips, contact our HR at oviya@thikseservices.onmicrosoft.com.',
    'Krishnakanth is the founder and director of the organization.',
    'Krishnakanth is the founder and director of the organization.',
    'The organization started in 2024.',
    'I am a Thikse chatbot, so I do not have an age.',
    'The weather today is sunny with a chance of clouds.',
    'Goodbye!',
    'Goodbye!',
]
 
# Define corpus for drill-down responses
career_corpus = [
    'What career opportunities are available?',
    'IT, Non-IT, Data Entry, full stack developer, Front-End, Back-End, AI Developer, Python, JavaScript, HR, Digital Marketing',
    'How can I apply for a job?',
    'Do you offer internships?',
]
 
career_responses = [
    'We have various career opportunities available. Could you specify your field of interest?',
    'Send your Resume to our HR: oviya@thikseservices.onmicrosoft.com.',
    'You can apply for a job by visiting our careers page on our website.',
    'Yes, we offer internships. For more details, please visit our careers page.',
]
 
organization_corpus = [
    'what are the service Do you provide ',
    'what is organization name',
    'what is your company name',
    'Who is your co-founder?',
    'co-founder name',
    'who is your Vice President',
    'VP name',
    'Tell me more about your organization.',
    'Who are the key people in your organization?',
    'What are the values of your organization?',
]
 
organization_responses = [
    'We Provide various services: Gen AI Solution\nApp Development\nWeb Development\nCyber Security Services\nDigital Marketing Solutions\nNon IT Services',
    'Our Organization name is "Thikse Software Solution PVT LTD"',
    'Our Organization name is "Thikse Software Solution PVT LTD"',
    "Our Co-Founder is Mr.Shiva\nAs a co-founder of Thikse Software Solutions, I drive our vision forward by fostering innovation and collaboration. I spearhead strategic initiatives, nurture our team's talents, and cultivate a culture of excellence. Together with my co-founders, I am passionate about delivering exceptional solutions to our clients.",
    "Our Co-Founder is Mr.Shiva\nAs a co-founder of Thikse Software Solutions, I drive our vision forward by fostering innovation and collaboration. I spearhead strategic initiatives, nurture our team's talents, and cultivate a culture of excellence. Together with my co-founders, I am passionate about delivering exceptional solutions to our clients.",
    "Our Vice President is Mrs.Malvika\nAs the Chief Financial Officer, I prioritize my responsibility by placing significant value on the engagement of our people and fostering a motivating environment. I actively support my leadership team in their aspirations and endeavors to achieve targets through effective management of funds and transactions.",
    "Our Vice President is Mrs.Malvika\nAs the Chief Financial Officer, I prioritize my responsibility by placing significant value on the engagement of our people and fostering a motivating environment. I actively support my leadership team in their aspirations and endeavors to achieve targets through effective management of funds and transactions.",
    'Our Mission: At Thikse Software Solutions, we\'re on a mission to empower talent and exceed client expectations. By championing fresh perspectives and securing exciting projects, we\'re building a dynamic community where innovation flourishes and success is inevitable.\n Our Vision: At Thikse Software Solutions, we prioritize quality and timely delivery while fostering our team\'s growth. Our commitment to excellence extends beyond projects—we\'re constantly innovating, expanding, and improving. Join us in creating a future where success knows no bounds.',
    'Our key people include our founder and director Mr.Krishnakanth, along with other talented professionals.',
    'Our values include integrity, teamwork, and customer satisfaction.',
]
 
 
# Defining Flask routes
@app.route('/')
def index():
    return send_from_directory('.','index.html')
 
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    category = request.json['category']
    if category.lower() == 'general':
        response = generate_response(user_input, category, corpus, responses)
    elif category.lower() == 'career':
        response = generate_response(user_input, category, career_corpus, career_responses)
    elif category.lower() == 'organization':
        response = generate_response(user_input, category, organization_corpus, organization_responses)
    else:
        response = "I'm sorry, I don't understand that category."
    return jsonify({'message': response})
 
# Function to process user input and generate response
def generate_response(user_input, category, corpus, responses):
    # Define a similarity threshold
    SIMILARITY_THRESHOLD = 0.3
   
    # Combine user input with relevant corpus
    all_corpus = corpus + [user_input]
   
    # Tokenize and stem the words in the new dataset
    stemmer = PorterStemmer()
    corpus_tokens = [nltk.word_tokenize(sentence.lower()) for sentence in all_corpus]
    corpus_stemmed = [' '.join([stemmer.stem(token) for token in tokens]) for tokens in corpus_tokens]
   
    # Vectorize the input
    corpus_vectorized = vectorizer.transform(corpus_stemmed)
    user_input_vectorized = corpus_vectorized[-1]
   
    # Compute cosine similarities
    similarities = cosine_similarity(user_input_vectorized, corpus_vectorized[:-1])
   
    # Check if the maximum similarity is below the threshold
    if max(similarities[0]) < SIMILARITY_THRESHOLD:
        response = "I'm sorry, I don't understand that."
    else:
        # Predict the response using the model
        prediction = model.predict(user_input_vectorized)
        predicted_label = np.argmax(prediction)
       
        # Retrieve the response from the label encoder
        response = label_encoder.inverse_transform([predicted_label])[0]
    #Set time to delay bot rpl
    time.sleep(1)
 
    return response
 
if __name__ == '__main__':
    app.run(debug=True)