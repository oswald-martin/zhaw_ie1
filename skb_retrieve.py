from pathlib import Path
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import SnowballStemmer
from nltk import FreqDist
import numpy as np
import re
from csv import reader

PATH_DOCS = 'skb/documents'
PATH_QRIS = 'skb/queries'
PATH_UGRM = 'skb/unigram_freq.csv'
OUTFILE = 'skb/result_mix.txt'
STEMMING = True
STOPWORDS = True
UNIGRAM = True
NR_RESULTS = 1000

########################################################################################################################
# Initiate NLTK
########################################################################################################################

download('punkt')
download('stopwords')
stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

########################################################################################################################
# Initiate data-structures
########################################################################################################################

invindex = defaultdict(lambda: defaultdict(int))
non_invindex = defaultdict(lambda: defaultdict(int))
queries = defaultdict(lambda: defaultdict(int))
accu = defaultdict(lambda: defaultdict(int))
d_norm = defaultdict(int)
idf = defaultdict(int)
unigram = defaultdict(lambda: 1)

########################################################################################################################
# Read unigram freq
########################################################################################################################

with open(PATH_UGRM) as csvfile:
    for row in reader(csvfile):
        token = stemmer.stem(row[0]) if STEMMING else row[0]
        freq = int(row[1])
        unigram[token] += freq

########################################################################################################################
# Index documents
########################################################################################################################

for doc in Path(PATH_DOCS).iterdir():
    print(f'indexing: {doc}')
    doc_text = doc.read_text()
    # tokenize text
    tokens = word_tokenize(doc_text)
    # eliminate stopwords
    tokens = [word for word in tokens if word.casefold() not in stopwords] if STOPWORDS else tokens
    # stem words
    tokens = [stemmer.stem(token) for token in tokens] if STEMMING else tokens
    # count words
    freq_dist = FreqDist(tokens)
    # fill datastructures
    for token, freq in freq_dist.items():
        invindex[token][doc] += freq
        non_invindex[doc][token] += freq

########################################################################################################################
# Compute IDF and dNorm
########################################################################################################################

total_nr_documents = len(non_invindex.keys())

for doc, tokens in non_invindex.items():
    for token in tokens:
        if UNIGRAM:
            idf[token] = np.log((1+total_nr_documents)/(1+invindex[token][doc]))/unigram[token]
        else:
            idf[token] = np.log((1+total_nr_documents)/(1+invindex[token][doc]))
        d_norm[doc] += (non_invindex[doc][token] * idf[token])**2
    d_norm[doc] = np.sqrt(d_norm[doc])

########################################################################################################################
# Index queries
########################################################################################################################

for query in Path(PATH_QRIS).iterdir():
    print(f'indexing: {query}')
    query_text = query.read_text()
    # tokenize text
    tokens = word_tokenize(query_text)
    # eliminate stopwords
    tokens = [word for word in tokens if word.casefold() not in stopwords] if STOPWORDS else tokens
    # stem words
    tokens = [stemmer.stem(token) for token in tokens] if STEMMING else tokens
    # count words
    freq_dist = FreqDist(tokens)
    # fill datastructure
    for token, freq in freq_dist.items():
        queries[query][token] += freq

########################################################################################################################
# Fill accumulatior
########################################################################################################################

for query, tokens in queries.items():
    q_norm = 0
    for token in tokens:
        idf_token = idf[token] if idf[token] != 0 else np.log(total_nr_documents+1)
        b = queries[query][token] * idf_token
        q_norm += b**2
        if token in invindex.keys():
            for doc in invindex[token]:
                a = invindex[token][doc] * idf_token
                accu[query][doc] += a*b

    q_norm = np.sqrt(q_norm)
    for doc in accu[query].keys():
        accu[query][doc] = accu[query][doc] / (d_norm[doc] * q_norm)
    accu_sorted = sorted(accu[query].items(), key=lambda x: x[1], reverse=True)
    n = 0
    for res, rsv in accu_sorted:
        if n == NR_RESULTS:
            break
        else:
            q = re.findall(r'\d+', query.__str__())[0]
            d = re.findall(r'\d+', res.__str__())[0]
            print(f'query:{query} \t document: {res} \t rank: {n} \t rsv: {rsv}')
            with open(OUTFILE, 'a') as f:
                f.write(f'{q} Q0 {d} {n} {rsv} celery\n')
            n += 1
    print('\n')
