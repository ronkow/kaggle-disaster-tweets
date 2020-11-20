import re           # re.sub
import nltk


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


# GLOBAL CONSTANTS
PUNC = '''!?.,;:$&*^~_"`(){}[]/\|<>=%+-'''  # exclude # (for #user) and @ (for @hashtag) and ' (so that can't is not converted to cant)

STOP_WORDS = set(stopwords.words('english')) # returns a set of stop words
ADD_WORDS = {"i'm"}
STOP_WORDS = STOP_WORDS.union(ADD_WORDS)

"""
STOP_WORDS
{'themselves', 'she', 'then', "doesn't", 'because', 'for', 'while', 'yourself', "shouldn't", 'not', 'or', "it's", 'having', 'other', 'after', 'again', 'from', 'which', "hadn't", 'ma', 'couldn', 'very', 'her', 'herself', 'does', 't', 'the', 'their', 're', 'ourselves', 'further', 'been', 'yourselves', 'of', 'don', "isn't", 'now', "couldn't", 'ours', 'this', "needn't", 'whom', 'any', 'during', 'but', 's', 'them', 'and', 'is', "aren't", 'what', 'most', 'only', 'doesn', 'it', 'myself', 'ain', "won't", 'weren', 'we', 'theirs', 'same', 'that', "weren't", 'won', 'in', 'you', 'his', 'i', 've', "mightn't", 'itself', 'just', 'few', 'be', 'if', 'under', 'hadn', 'him', 'wasn', 'below', 'there', "hasn't", 'nor', 'were', 'here', 'are', 'some', 'no', 'before', 'he', "you're", 'its', 'have', 'at', 'mustn', 'didn', "that'll", 'off', 'should', 'wouldn', 'where', 'll', 'haven', 'into', 'as', 'our', 'your', "don't", 'until', 'how', "haven't", 'once', 'shan', 'mightn', "didn't", "mustn't", 'shouldn', 'had', "you'll", 'my', 'about', 'o', 'between', 'all', 'both', 'over', "you'd", 'isn', "shan't", 'hers', 'so', 'has', 'more', 'did', 'against', 'who', 'by', 'when', "should've", "you've", 'each', 'such', 'me', 'a', 'out', 'those', 'an', 'down', 'am', 'hasn', 'on', 'why', 'needn', 'with', 'was', "wasn't", "wouldn't", 'through', 'too', "i'm", 'will', "she's", 'they', 'y', 'own', 'can', 'himself', 'm', 'these', 'than', 'doing', 'above', 'd', 'up', 'being', 'do', 'yours', 'to', 'aren'}
"""

def clean_doc_text(doc):
    """
    ARGUMENT: text (string)
    RETURN: list of tokens
    """
    doc = doc.replace('...',' ... ')  # to avoid converting abc...xyz to abcxyz
    doc = doc.replace("'",' ')        # to convert "can't" to "can" and "t"
    
    for p in PUNC:
        doc = doc.replace(p,'')
  
    tokens = doc.split()                                  # returns a list of tokens
    tokens = [w.lower() for w in tokens]                  # convert all letters to lower case  
    tokens = [w for w in tokens if not w in STOP_WORDS]   # exclude stop words
    
    tokens = [w for w in tokens if not w.isdigit()]       # exclude all numbers, but include words with numbers, such as abc12

    tokens = [porter.stem(w) for w in tokens]             # stemming
    tokens = [w for w in tokens if len(w)>=2]             # include only words with length >= 2
    
    return tokens                                         # list of tokens


def clean_doc_text_bow(tokens, token_list):
    tokens = [w for w in tokens if w in token_list]       # include only tokens in selected bag of words
    return tokens                                         # list of tokens


def clean_doc_keyword(doc):
    """
    ARGUMENT: text (string)
    RETURN: list of tokens
    """
    for p in PUNC:
        doc = doc.replace(p,'')
  
    tokens = doc.split()                                 # returns a list of tokens
    tokens = [w.lower() for w in tokens]                 # convert all characters to lower case  
    tokens = [re.sub(r'\d+', '', w) for w in tokens]    # sub numbers with underscore
    tokens = ['kw_'+ w for w in tokens]
    return tokens                                        # list of tokens


def list_of_token_lists_text(text_list, token_list):
    text_token_list = []
    
    for x in text_list:
        y = clean_doc_text(x)
        y = clean_doc_text_bow(y, token_list)
        text_token_list.append(y)
    return text_token_list



def list_of_token_lists_keyword(keyword_list):
    keyword_token_list = []
    
    for x in keyword_list:
        y = clean_doc_keyword(x)
        keyword_token_list.append(y)
    return keyword_token_list
