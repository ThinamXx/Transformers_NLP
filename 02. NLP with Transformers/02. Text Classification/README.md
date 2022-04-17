## **Text Classification**

The [**Text Classification**](https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/02.%20Text%20Classification/Text%20Classification.ipynb) notebook contains information and implementation of Character Tokenization, Word Tokenization, Subword Tokenization, Feature Extraction, Training Logistic Regression & Hidden States, One Hot Encoding, Fine-Tuning Transformers, Confusion Matrix, Performance Metrics and Error Analysis.

**Note:**
- 📝[**Text Classification**](https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/02.%20Text%20Classification/Text%20Classification.ipynb)

**Class Imbalance**
- Ways to deal with imbalanced data:
  - Randomly oversample the minority class.
  - Randomly undersample the majority class.
  - Gather more labeled data from the underrepresented classes.

**Text to Tokens**
- **Tokenization** is the step of breaking down a string into the atomic units used in the model. **Numericalization** is the process of converting tokens into integer.

**Subword Tokenization**
- Subword tokenization combines the best aspects of character and word tokenization. It splits rare words into smaller units to allow the model to deal with complex words and misspellings and it keeps frequent words as unique entities so that the length of inputs are kept to a manageable size.

**DistilBERT**
- The main advantage of this model is that it achieves comparable performance to BERT, while being significantly smaller and more efficient.
