from pathlib import Path
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import snowball

PATH_DOCS = 'documents'
PATH_QRIS = 'queries'
STEMMING = True
STOPWORDS = True

########################################################################################################################
# Load NLTK packages
########################################################################################################################

download('punkt')
download('stopwords')

########################################################################################################################
# Initiate data-structures
########################################################################################################################

invindex = defaultdict(lambda: defaultdict(int))
non_index = defaultdict(lambda: defaultdict(int))
queries = defaultdict(lambda: defaultdict(int))

########################################################################################################################
# Load Stopwords
########################################################################################################################

stopwords = set(stopwords.words('english'))

########################################################################################################################
# Read documents
########################################################################################################################

for doc in Path(PATH_DOCS).iterdir():
    doc_text = doc.read_text()
    # tokenize text
    tokens = word_tokenize(doc_text)
    # eliminate stopwords
    tokens = [word for word in tokens if word.casefold() not in stopwords] if STOPWORDS else tokens
    print(tokens)
