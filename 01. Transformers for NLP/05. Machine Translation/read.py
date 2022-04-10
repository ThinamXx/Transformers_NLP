#@ PROCESSING THE DATASET:
import pickle
from pickle import dump

import re 
import string
import unicodedata

#@ LOADING INTO MEMORY:
def load_doc(filename):                                 # Defining function.
    file = open(filename, mode="rt", encoding="utf-8")  # Opening file.
    text = file.read()                                  # Reading file.
    file.close()                                        # Closing file.
    return text                                         # Getting all texts.

#@ SPLITTING INTO SENTENCES:
def to_sentence(doc):                                   # Defining function.
    return doc.strip().split("\n")                      # Splitting into sentences. 

#@ SHORTEST AND LONGEST SENTENCES:
def sentence_lengths(sentences):                        # Defining function.
    lengths = [len(s.split()) for s in sentences]       # Lengths of sentence.
    return min(lengths), max(lengths)                   # Getting minimum and maximum.

#@ CLEANING SENTENCES:
def clean_lines(lines):                                            # Defining function.
    cleaned = list()                                               # Initialization. 
    re_print = re.compile('[^%s]' % re.escape(string.printable))   # Preparing regex.
    table = str.maketrans('', '', string.punctuation)              # Removing punctuation. 
    for line in lines:
        line = unicodedata.normalize(
            "NFD", line).encode("ascii", "ignore")                 # Normalizing unicode.
        line = line.decode("UTF-8")
        line = line.split()                                        # Tokenization.
        line = [word.lower() for word in line]                     # Lower case.
        line = [word.translate(table) for word in line]            # Remove punctuation.
        line = [re_print.sub('', w) for w in line]                 # Remove non-printable.
        line = [word for word in line if word.isalpha()]           # Remove numbers.
        cleaned.append(' '.join(line))                             # Storing as string.
    return cleaned 

#@ LOADING ENGLISH DATA:
path_file_en = "/content/drive/MyDrive/Data/europarl-v7.fr-en.en"  # English data.
doc = load_doc(path_file_en)                                       # Loading.
sentences = to_sentence(doc)                                       # Splitting into sentences.
minlen, maxlen = sentence_lengths(sentences)                       # Shortest and longest.
print("English data: sentences=%d, min=%d, max=%d" %(
    len(sentences), minlen, maxlen))
cleanf = clean_lines(sentences)                                    # Cleaning sentences.
filename = "English.pkl"                                           # Initialization.
outfile = open(filename, "wb")
pickle.dump(cleanf, outfile)                                       # Storing.
outfile.close()                                                    # Closing.
print(filename, " saved")                                          # Inspection. 

#@ LOADING FRENCH DATA:
path_file_fr = "/content/drive/MyDrive/Data/europarl-v7.fr-en.fr"  # French data.
doc = load_doc(path_file_fr)                                       # Loading.
sentences = to_sentence(doc)                                       # Splitting into sentences.
minlen, maxlen = sentence_lengths(sentences)                       # Shortest and longest.
print("French data: sentences=%d, min=%d, max=%d" %(
    len(sentences), minlen, maxlen))
cleanf = clean_lines(sentences)                                    # Cleaning sentences.
filename = "French.pkl"                                            # Initialization.
outfile = open(filename, "wb")
pickle.dump(cleanf, outfile)                                       # Storing.
outfile.close()                                                    # Closing.
print(filename, " saved")                                          # Inspection. 