## **Transformer Anatomy**

The [**Transformer Anatomy**](https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/03.%20Transformer%20Anatomy/TransformerAnatomy.ipynb) notebook contains information and implementation of The Encoder Transformer Architecture, Scaled Dot Product Attention, Tokenization, Embedding Layer, Softmax Activation Function, Multi-head Attention, Feed-Forward Layer, Layer Normalization and The Decoder Transformer. 

**Note:**
  - 📝[**Transformer Anatomy**](https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/03.%20Transformer%20Anatomy/TransformerAnatomy.ipynb)

**The Encoder**
- The numerical representation computed for a given token in encoder only transformer architecture depends both on the left or before the token and the right or after the token contexts which is called bidirectional attention.

**The Decoder**
- The numerical representation computed for a given token in decoder only transformer architecture depends only on the left context which is called autoregressive attention.

**Feed-Forward Layer**
- Skip connections pass a tensor to the next layer of the model without processing and add it to the processed tensor.

**Layer Normalization**
- Layer normalization normalizes each input in the batch to have zero mean and unity in variance.
