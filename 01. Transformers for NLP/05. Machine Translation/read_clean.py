#@ PREPROCESSING THE DATASET:

import pickle
from pickle import load
from pickle import dump
from collections import Counter

#@ LOADING CLEAN DATASET:
def load_clean_sentences(filename):         # Defining function.
    return load(open(filename, "rb"))       # Loading clean dataset.

#@ SAVING CLEAN DATASET:
def save_clean_sentences(sentences, filename):      # Defining function. 
    dump(sentences, open(filename, "wb"))           # Saving dataset. 
    print("Saved: %s" % filename)                   # Inspection.

#@ CREATING SEQUENCE TABLE: VOCABULARY:
def to_vocab(lines):                                # Defining function. 
    vocab = Counter()                               # Initializing counter.
    for line in lines:
        tokens = line.split()                       # Tokenization.
        vocab.update(tokens)                        # Updating.
    return vocab                                    # Getting vocabulary.

#@ PREPROCESSING VOCABULARY:
def trim_vocab(vocab, min_occurance):                               # Defining function.
    tokens = [k for k,c in vocab.items() if c >= min_occurance]     # Trimming vocabulary tokens.
    return set(tokens)                                              # Getting tokens.

#@ PROCESSING OOV WORDS:
def update_dataset(lines, vocab):                                   # Defining function.
    new_lines = list()                                              # Initialization. 
    for line in lines:
        new_tokens = list()                                         # Initialization.
        for token in line.split():
            if token in vocab:
                new_tokens.append(token)
            else:
                new_tokens.append("unk")                            # Adding unknown tokens.
        new_line = ' '.join(new_tokens)
        new_lines.append(new_line)
    return new_lines                                                # Getting updated lines.

#@ LOADING ENGLISH DATASET:
filename = "English.pkl"                                            # Initialization.
lines = load_clean_sentences(filename)                              # Loading.
vocab = to_vocab(lines)                                             # Initializing vocabulary. 
print("English vocabulary: %d" % len(vocab))                        # Inspection.
vocab = trim_vocab(vocab, 5)                                        # Reducing vocabulary.
print("New English vocabulary: %d" % len(vocab))                    # Inspection. 
lines = update_dataset(lines, vocab)                                # Processing OOV.
filename = "english_vocab.pkl"                                      # Initialization. 
save_clean_sentences(lines, filename)
for i in range(10):
    print("line", i, ":", lines[i])                                 # Inspection.

#@ LOADING FRENCH DATASET:
filename = "French.pkl"                                             # Initialization.
lines = load_clean_sentences(filename)                              # Loading.
vocab = to_vocab(lines)                                             # Initializing vocabulary. 
print("French vocabulary: %d" % len(vocab))                         # Inspection.
vocab = trim_vocab(vocab, 5)                                        # Reducing vocabulary.
print("New French vocabulary: %d" % len(vocab))                     # Inspection. 
lines = update_dataset(lines, vocab)                                # Processing OOV.
filename = "french_vocab.pkl"                                       # Initialization. 
save_clean_sentences(lines, filename)
for i in range(10):
    print("line", i, ":", lines[i])                                 # Inspection.