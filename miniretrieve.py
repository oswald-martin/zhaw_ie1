from pathlib import Path
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import SnowballStemmer
from nltk import FreqDist

PATH_DOCS = 'documents'
PATH_QRIS = 'queries'
STEMMING = True
STOPWORDS = True

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

########################################################################################################################
# Index documents
########################################################################################################################

for doc in Path(PATH_DOCS).iterdir():
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
# Index queries
########################################################################################################################

for query in Path(PATH_QRIS).iterdir():
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
