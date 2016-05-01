#coding: utf-8

import sys
import resource
import logging
import time
import unicodedata
from datetime import timedelta
from optparse import OptionParser
import scipy as sp
import os
import csv
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import operator
import html2text
from simple_ga import Simple_GA
from collections import defaultdict

def getMoodleData(mypath="./",filename="resposta.txt"):
    data = []
    sistema = ["__pycache__"]
    arquivos = [x for x in os.listdir(mypath) if x not in sistema]
    subpastas = [os.path.join(mypath,subdir) for subdir in arquivos if os.path.isdir(os.path.join(mypath,subdir))]
    listaux = [(x[2:].split("-")) for x in subpastas]
    listaux=[int(x) for x,y in listaux]
    listaux = [i[0] for i in sorted(enumerate(listaux), key=lambda x:x[1])]
    subpastas = [subpastas[x] for x in listaux]
    for sub in subpastas:
        with open(sub+"/"+filename,"r", encoding='iso-8859-1') as txt:
            auxfolder = sub[(sub.rfind("/")+1): ]
            auxsub = html2text.html2text(txt.read(), "iso-8859-1")
            if auxsub != "":
                data.append([auxfolder,auxsub])
    return data, len(data)

def getCSVData(csvfile, header=True):
    with open(csvfile) as csvready:
        filereader = csv.reader(csvready, delimiter='\t')
        data = []
        for row in filereader:
            data.append('\t'.join(row))
        if header:
            return data[0], data[1:]
        else:
            return data

def setCSVData(csvfile, rows):
    with open(csvfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            csvwriter.writerow(r)

def std_text(row):
    row = re.sub("<.*?>", "", row)
    tmp = [p.lower() for p in row.split() if not p.lower() in stopwords.words("portuguese")]
    aux = []
    for s in tmp:
        aux.append(re.sub(r'\W+', '', remove_accents(s)))
    return aux

def remove_accents(string):
    string = unicodedata.normalize('NFD', string)
    return u''.join(ch for ch in string if unicodedata.category(ch) != 'Mn')

def stemmWord(word):
    ptbrstemm = nltk.SnowballStemmer("portuguese",)
    return ptbrstemm.stem(word)

def stemmTxt(txt):
    ptbrstemm = nltk.SnowballStemmer("portuguese",)
    tempcorpus =[]
    for sbm in txt:
        submission = []
        for w in sbm:
            #submission.append(ptbrstemm.stem(w))
            submission.append(w)
        tempcorpus.append(' '.join(submission))
    return tempcorpus

def save_features(fname, feature_names):
    features = [word for rep in feature_names for word, grade, weight in rep]
    grades = [grade for rep in feature_names for word, grade, weight in rep]
    weight = [weight for rep in feature_names for word, grade, weight in rep]
    arr = zip(features, weight, grades)
    arr = sorted(arr, key=operator.itemgetter(1))
    with open(fname, 'w') as arq:
        for w, h, g in reversed(arr):
            arq.write(str(w)+"\t"+str(g)+"\t"+str(h)+"\n")

def readDoc(nome_res):
    with open(nome_res,"r", encoding='iso-8859-1') as txt:
        return txt.read()

def makeHTMLReport(student_data, report, filename="report.html", mypath="./"):
    tokenizer = re.compile('\w+')
    wordset = []
    for word, grade, weight in report[0]:
        if wordset != [] or stemmWord(remove_accents(word.lower())) not in [word_ for word_, value_, grade_ in wordset]:
            wordset.append((stemmWord(word),weight, grade))

    for idx, data in enumerate(student_data):
        folder, grade, answer = data
        text = ""
        for w in tokenizer.findall(answer):
            if stemmWord(remove_accents(w.lower())) in [word_ for word_, value_, grade_ in wordset]:
                index = [word_ for word_, value_, grade_ in wordset].index(stemmWord(remove_accents(w.lower())))
                intensity = int((1-round((wordset[index][1]*wordset[index][2])/2,1))*10)
                if intensity > 9:
                    intensity = 9
                text+="<b><font color='#"+(str(intensity)*6)+"'>"+w+" </font></b>"
            else:
                text+=w+" "
        html_str = """
<html>
    <head>
        <meta charset="ISO-8859-1" />
    </head>
    <body>
        <p>"""+text+"""</p>
    </body>
</html>
"""
        with open(mypath+folder+"/"+filename,"w", encoding='iso-8859-1') as html: 
            html.write(html_str)
    return wordset


def reportweights(report, vector, train):
    maxgrade = max(np.unique([float(x) for x in train[:, 1]]))
    mingrade = min(np.unique([float(x) for x in train[:, 1]]))
    weightsArr = np.zeros(np.shape(vector)[1])
    words_weights = []
    wordset = defaultdict(set)
    w_weigths = []
    for i, words, c in report:
        for w in words.split(" "):
            wordset[w].add(i)
    maxlen = max([len(wordset[w]) for w in wordset.keys()])
    
    if float(maxgrade - mingrade) == 0:
        norm_grade = 1
    else:
        norm_grade = float(maxgrade - mingrade)/ float(maxgrade - mingrade)

    for w in wordset.keys():
        w_weigths.append((w, len(wordset[w])/maxlen, norm_grade))
    for word, weight, grade in w_weigths:
        for idx in wordset[w]:
            if weightsArr[idx] == 0:
                weightsArr[idx] = (weight * norm_grade)/2
    words_weights.append(w_weigths)
    weightsArr = [(1+val) for val in weightsArr]
    xInd = yInd = range(len(weightsArr))
    features_weights = sp.sparse.csr_matrix((weightsArr, (xInd, yInd)))
    vector = np.multiply(vector, features_weights)
    return vector, words_weights

class ElapsedFormatter():
    
    def __init__(self):
        self.start_time = time.time()
    
    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        #using timedelta here for convenient default formatting
        elapsed = timedelta(seconds = elapsed_seconds)
        return "[%s][RAM: %.2f MB] %s" % (str(elapsed)[:-3], (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), record.getMessage())

def main():
    parser = OptionParser(usage="%prog [options] <datasetdir>")
    parser.add_option("-e", "--encoding", dest="encoding", default="iso-8859-1", help="Dataset encoding")

    if sys.stdout.encoding == None:
        print("Fixing stdout encoding...")
        import codecs
        import locale
        # Wrap sys.stdout into a StreamWriter to allow writing unicode.
        sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout)

    (options, args) = parser.parse_args()

    #add custom formatter to root logger for simple demonstration
    handler = logging.StreamHandler()
    handler.setFormatter(ElapsedFormatter())
    logging.getLogger().addHandler(handler)

    log = logging.getLogger('main')
    log.setLevel(logging.DEBUG)

    '''Text Processing Start - Code'''

    student = []
    submit = []
    
    log.info("[Process] Loading database")
    
    db, tam = getMoodleData()
    train_db = getCSVData("notastreino.csv",False)

    log.info("[Process] Std. database [Stemming / Stopwords Rem. / Punctuation Rem.]")

    for subm in db:
        student.append(subm[0])
        submit.append(std_text(subm[1]))
    
    txtsubmission = stemmTxt(submit)
    
    train_data = np.array([(y, float(x.split(";")[1])) for y, x in enumerate(train_db) if float(x.split(";")[1]) > -1])
    log.info(train_data)
    n_clusters = np.shape(np.unique(train_data[:, 1]))[0]

    log.info("[Process] TF-IDF")
    
    vectorizer = TfidfVectorizer(ngram_range=(2,5))
    vcnt = vectorizer.fit_transform(txtsubmission)
    words = vectorizer.get_feature_names()

    trainX = vcnt

    report = []

    log.info(np.unique([float(x) for x in train_data[:, 1]]))
    topfeatures =  []
    
    tag = [x for x in reversed(sorted(np.unique([float(x) for x in train_data[:, 1]])))][0]
    log.info("[Process] Starting GA for "+str(tag)+" Data")
    ids = [int(x2[0]) for idx, x2 in enumerate(train_data) if x2[1] == tag]
    log.info(ids)
    if len(ids) > 1:
        get_ids = [idx for idx, x  in sorted(enumerate(vcnt[ids].sum(axis=1)), key=lambda x:x[1])]
        log.info(get_ids)
        ga_fselection = Simple_GA(vcnt[ids[get_ids[0]]].todense(), 100, 500,vcnt[ids])
        topfeatures = ga_fselection.executeGA()

    else: 
        topfeatures = []
        for x in range(np.shape(vcnt[ids])[1]):
            if x > 0:
                topfeatures.append(1)
            else:
                topfeatures.append(0)
    report = [(idx,words[idx], value) for idx, value in enumerate([value_ for value_ in topfeatures]) if value > 0]

    resultado = []

    vcnt, w_weights = reportweights(report, vcnt, train_data)
    
    log.info("[Process] Generating Report")
    resultado = []
    for value in range(tam):
        resultado.append((value, np.unique([float(x) for x in train_data[:, 1]])[0]))
    
    data_report = [(db[i][0], g, db[i][1]) for i, g in resultado]
    
    data_report = [data_report[i[0]] for i in reversed(sorted(enumerate([float(grade) for idx, grade, answer in data_report]), key=lambda x:x[1]))]

    makeHTMLReport(data_report, w_weights)

    save_features("wweights.txt", w_weights)

if __name__ == "__main__":
    main()