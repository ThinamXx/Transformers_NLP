## **Text Generation**

The [**Text Generation**](https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/05.%20Text%20Generation/Text%20Generation.ipynb) notebook contains information and implementation of Greedy Search Decoding, Beam Search Decoding, Sampling Methods, Top-k and Nucleus Sampling.

**Note:**
  - 📝[**Text Generation**](https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/05.%20Text%20Generation/Text%20Generation.ipynb)


**Greedy Search Decoding**
- The simplest decoding method to get discrete tokens from a model's continuous output is to greedily select the token with the highest probability at each timestamp.

**Beam Search Decoding**
- Instead of decoding the token with highest probability at each step, beam search keeps track of the top-b most probable next tokens, where b referred to as the number of beams or partial hypotheses.
