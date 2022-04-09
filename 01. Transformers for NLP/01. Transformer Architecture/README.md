## **Model Architecture of Transformer**

The [**Positional Encoding**](https://github.com/ThinamXx/Transformers_NLP/blob/main/01.%20Transformers%20for%20NLP/01.%20Transformer%20Architecture/PositionalEncoding_M.ipynb) notebook contains information about Positional Encoding, Word Embeddings and Positional Vectors. 

The [**Transformer Architecture**](https://github.com/ThinamXx/Transformers_NLP/blob/main/01.%20Transformers%20for%20NLP/01.%20Transformer%20Architecture/TransformerArchitecture_T.ipynb) notebook contains information about Input Embedding, Multi-head Attention, Weight Matrices, Matrix Multiplication and Attention Representations. 

**Note:**
  - 📝[**Positional Encoding**](https://github.com/ThinamXx/Transformers_NLP/blob/main/01.%20Transformers%20for%20NLP/01.%20Transformer%20Architecture/PositionalEncoding_M.ipynb)
  - 📝[**Transformer Architecture**](https://github.com/ThinamXx/Transformers_NLP/blob/main/01.%20Transformers%20for%20NLP/01.%20Transformer%20Architecture/TransformerArchitecture_T.ipynb)

**Input Embedding**
- The input embedding sub-layer converts the input tokens to vectors of dimension: 512 using learned embeddings in the original Transformer model. Cosine similarity uses Euclidean (L2) norm to create vectors in a unit sphere.

**Positional Encoding**
- The idea is to add a positional encoding value to the input embedding instead of having additional vectors to describe the position of the token in a sequence.
