from flask import Flask,request,render_template
import pickle

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
import numpy as np
#import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
#from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/main')
def main():
   return render_template('main.html')
@app.route('/feed')
def feed():
   return render_template('feedback.html')
@app.route('/contact')
def contact():
   return render_template('contact.html')

@app.route('/pre',methods=['POST'])
def pre():
    #val=[x for x in request.form.values()]
    val=[x for x in request.form["input"]]
    li=[x for x in request.form["lines"]]
    no_of_lines=int(li[0])
    text="".join(x for x in val)
   
    sentences=sent_tokenize(text)
    sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]
    stop_words = stopwords.words('english')

    sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    sentence_embeddings = []
    for i in sentence_tokens:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i])/(len(i)+0.001)
        else:
            v = np.zeros((100,))
        sentence_embeddings.append(np.array(v))

    sentence_embeddings=np.array(sentence_embeddings)
    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i,row_embedding in enumerate(sentence_embeddings):
        for j,column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences)}

    top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:no_of_lines])
    need=''
    for sent in sentences:
        if sent in top.keys():
            #print(sent)
            need+=sent
    
    return render_template('main.html',output="{0}".format(need))




if __name__ == "__main__":
    app.run(debug=True)