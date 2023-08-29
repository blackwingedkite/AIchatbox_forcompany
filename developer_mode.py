"""
Develop Mode可以接觸資料庫的後端，利用Chorma存檔，並且可以改變不同的embeddings.
"""
from flask import Flask, render_template, request, redirect
import csv
import os
# from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from markupsafe import escape

import requests


app = Flask(__name__, '/static')
# _ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = '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'
llm = ChatOpenAI(temperature=0.7)

file = 'static/data.csv'
loader = CSVLoader(file_path=file, encoding="utf-8")
docs = loader.load()
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings, persist_directory="static/")


memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    memory = memory
)
template = """請假設您是全知的濾能公司AI客服。請您根據過往對話和問答資料，進行微調，以更好的回覆這個用戶的問題，並不要進行任何隱瞞。若內容中出現濾能相關資料，則隱晦的推薦濾能股份有限公司，若無，則不要提到該公司。請將所有前言改寫為「您好，我是濾能的AI客服。
        過往對話: {history}
        問答資料: {input}
        濾能公司AI客服:"""
        




PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="濾能公司AI客服"),
)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    memory = memory
)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'next_question' in request.form:  # 如果存在'next_question'参数，則表示用戶提交了下一個問題
            query = request.form['text_input']
            # 繼續對話
            docs_similar = db.similarity_search(query)
            db_similar = DocArrayInMemorySearch.from_documents(docs_similar, embeddings)
            retriever = db_similar.as_retriever()
            qa_stuff = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever, verbose=True)
            response = qa_stuff.run(query)
            gptresponse = conversation.predict(input="客戶問題: "+query+"\n"+"回答: "+response)
        else:
            query = request.form['text_input']
            # 首次對話
            docs_similar = db.similarity_search(query)
            db_similar = DocArrayInMemorySearch.from_documents(docs_similar, embeddings)
            retriever = db_similar.as_retriever()
            qa_stuff = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever, verbose=True)
            response = qa_stuff.run(query)
            gptresponse = conversation.predict(input="客戶問題: "+query+"\n"+"回答: "+response)
        return render_template('index.html', prediction=response, sentence=query, gptprediction=gptresponse)
    refresh = request.args.get('refresh')
    if refresh == 'true':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    sentence = request.form['sentence']
    gptprediction = request.form['gptprediction']

    with open(file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sentence, gptprediction])

    docs = loader.load()
    db = Chroma.from_documents(docs, embeddings, persist_directory="static/")
    db.persist()
    return render_template('index.html')


@app.route('/submit_prediction', methods=['POST'])
def submit_prediction():
    sentence = request.form['sentence']
    prediction = request.form['prediction']

    with open(file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sentence, prediction])
    docs = loader.load()
    db = Chroma.from_documents(docs, embeddings, persist_directory="static/")
    db.persist()
    return render_template('index.html')


@app.route('/custom_input', methods=['POST'])
def custom_input():
    sentence = request.form.get('sentence')
    user_input = request.form.get('user_input')

    with open(file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sentence, user_input])

    docs = loader.load()
    db = Chroma.from_documents(docs, embeddings, persist_directory="static/")
    db.persist()
    return render_template('index.html')


@app.route('/discard', methods=['POST'])
def discard():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

