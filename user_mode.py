"""
Develop Mode可以接觸資料庫的後端，利用Chorma存檔，並且可以改變不同的embeddings.
"""
from flask import Flask, render_template, request, redirect
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import requests

app = Flask(__name__, '/static')
_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = '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'

embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="static/", embedding_function=embeddings)

@app.route('/', methods=['GET', 'POST'])
def usermode():
    if request.method == 'POST':
        query = request.form['text_input']
        docs_similar = db.similarity_search(query)
        db_similar = DocArrayInMemorySearch.from_documents(docs_similar, embeddings)
        retriever = db_similar.as_retriever()
        llm = ChatOpenAI(temperature=0.7)
        qa_stuff = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, verbose=True)
        response = qa_stuff.run(query)

        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": "請假設您是全知的濾能公司的客服。請您將回答進行微調，以更好的回覆這個用戶的問題，並不要進行任何隱瞞。若內容中出現濾能相關資料，則隱晦的推薦濾能股份有限公司，若無，則不要提到該公司。請不要寫任何的前言。"},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ],
            "model": "gpt-3.5-turbo"
        }

        gptresponse = requests.post(api_url, json=payload, headers=headers)
        response_data = gptresponse.json()
        try:
            chatgpt_answer = response_data["choices"][0]["message"]["content"]
        except:
            chatgpt_answer = response_data['error']['message']
        return render_template('usermode.html', prediction=response, sentence=query, gptprediction=chatgpt_answer)
    refresh = request.args.get('refresh')
    if refresh == 'true':
        return render_template('usermode.html')
    return render_template('usermode.html')

@app.route('/discard', methods=['POST'])
def discard():
    return render_template('usermode.html')


if __name__ == '__main__':
    app.run(debug=True)

