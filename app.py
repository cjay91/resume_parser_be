#C:\flask_dev\flaskreact\app.py
from flask import Flask, json, request, jsonify
import urllib.request
from werkzeug.utils import secure_filename #pip install Werkzeug
from flask_cors import CORS #ModuleNotFoundError: No module named 'flask_cors' = pip install Flask-Cors
import fitz
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
from openai import OpenAI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json



nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app, supports_credentials=True)
 
app.secret_key = "caircocoders-ednalan"
  
UPLOAD_FOLDER = 'static/uploads'
JD_UPLOAD_FOLDER = 'static/uploads/jd'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JD_UPLOAD_FOLDER'] = JD_UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
ALLOWED_EXTENSIONS = set(['csv', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
  
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  
@app.route('/')
def main():
    return 'Homepage'
  
@app.route('/upload', methods=['GET','POST'])
def upload_file():

    # filename = ''
    # check if the post request has the file part
    print(request.files)
    if 'files[]' not in request.files:
        resp = jsonify({
            "message": 'No file part in the request',
            "status": 'failed'
        })
        resp.status_code = 400
        return resp
  
    files = request.files.getlist('files[]')
      
    errors = {}
    success = False
    file_content = {}

    for file in files:      
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            directory = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(directory)

            doc = fitz.open(directory)
            text_list = []

            text = ''
            for page in doc:
                text += page.get_text()
            text_list.append(text)
            file_content[filename] = text_list
            # Create a DataFrame from the file_content dictionary
            

            success = True
        else:
            resp = jsonify({
                "message": 'File type is not allowed',
                "status": 'failed'
            })
            return resp
        
    column_names = ['Text']
    df = pd.DataFrame.from_dict(file_content, orient='index', columns=column_names)
    excel_filename = 'output_1.xlsx'
    df.to_excel(os.path.join(app.config['UPLOAD_FOLDER'], excel_filename), index_label='Filename')
         
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        errors['status'] = 'failed'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({
            "message": 'Files successfully uploaded',
            "status": 'successs'
        })
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 

  
@app.route('/upload-jd', methods=['GET','POST'])
def upload_jd_file():

    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({
            "message": 'No file part in the request',
            "status": 'failed'
        })
        resp.status_code = 400
        return resp
  
    files = request.files.getlist('files[]')
    
    errors = {}
    success = False

    filename = secure_filename(files[0].filename)
    directory = os.path.join(app.config['JD_UPLOAD_FOLDER'], filename)
    files[0].save(directory)

    doc = fitz.open(directory)
    text_list = ''

    text = ''
    for page in doc:
        text += page.get_text()
    text_list += text
      
    # keyword_list = get_jd_keywords(text_list)
   
    with open('some_file.txt', 'w') as f:
        f.write(text_list)

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        errors['status'] = 'failed'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({
            "message": 'Files successfully uploaded',
            "status": 'successs'
        })
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def get_jd_keywords(text):

    myMessages = []
    myMessages.append(
    {"role": "system", "content": "You are an expert in your domain. Please provide your expert and user-friendly response based on the context provided. You should aim to provide a clear, concise, and accurate response including contact details. Also rank the CVs based on how suitable for the job description"})

    myMessages.append(
    {"role": "user", "content": "cv_list:\n\n give the keywords in the job description{} as a python list.Only give maximum 3 words per keyword.Example list is like '['Java','Python']'. Only output should be the list without any text before or after the list".format(text)})

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=myMessages,
        stream=False,
    )

    return response.choices[0].message.content.strip()

def calculate_score(cv_data):
        # Score each candidate
    cv_data['Text'] = cv_data['Text'].astype(str)
    cv_texts = cv_data['Text'].tolist()

    scores = []

    # Open the file in read mode
    with open('some_file.txt', 'r') as file:
    # Read the entire content into a variable
        job_description = file.read()

    documents = cv_texts + [job_description]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
    # Round the scores to two decimal places
    scores = [(round(score, 2) * 100) for score in cosine_similarities.flatten()]

    # Add an auto-incrementing 'id' column
    cv_data['id'] = range(1, len(cv_data) + 1)
    
    # Add rounded scores to the DataFrame
    cv_data['Score'] = scores


    output_file_path = 'cv_data_with_scores.csv'  # Update this path if needed
    cv_data.to_csv(output_file_path, index=False)
    
    print(f"Updated CSV saved to {output_file_path}")

@app.route('/get_data', methods=['GET','POST'])
def get_data():
    data = pd.read_excel('/home/chathura/Development/ResumeParser/react_flask_app/backend/static/uploads/output_1.xlsx')
    calculate_score(data)
    cv_score_data = pd.read_csv('cv_data_with_scores.csv')
    cv_score_data = cv_score_data.to_json(orient='records', lines=True).splitlines()

    return jsonify(cv_score_data)



@app.route('/chat-with-csv', methods=['GET','POST'])
def chat_with_csv():
    # input = request.json.get('Title', "")
    input = request.json.get('post', {}).get('title', '')

    embedding_function = OpenAIEmbeddings()

    loader = CSVLoader("cv_data_with_scores.csv", encoding="utf8")
    documents = loader.load()

    db = Chroma.from_documents(documents, embedding_function)
    retriever = db.as_retriever()

    template = """Answer the question based only on the following context:
    {context} This has a data frame with names of the candidates, 
    the text from their CVs and the score which indicates its similarity to the given job description. 
    Based on these information answer the below question.

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(input)

if __name__ == '__main__':
    app.run(debug=True)