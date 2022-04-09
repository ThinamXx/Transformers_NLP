## **Fine-Tuning BERT Models**

The [**BERT**](https://github.com/ThinamXx/Transformers_NLP/blob/main/01.%20Transformers%20for%20NLP/02.%20Fine-Tuning%20BERT%20Model/BERT.ipynb) notebook contains information about BERT Architecture, Processing Dataset, BERT Tokenizer, Attention Masks, BERT Model Configuration, Optimizer Grouped Parameters, Initializing Hyperparameters, Tranining & Evaluation and Matthews Correlation Coefficient. 

**Note:**
- [**BERT**](https://github.com/ThinamXx/Transformers_NLP/blob/main/01.%20Transformers%20for%20NLP/02.%20Fine-Tuning%20BERT%20Model/BERT.ipynb)

**Optimizer Grouped Parameters**
- Fine-tuning a model begins with initializing the pretrained model parameter values. We shouldn't apply weight decay to parameters: bias, gamma, beta.

**Initializing Hyperparameters**
- The learning rate and warm-up rate should be set to a very small value early in the optimization phase and gradually increase after a certain number of iterations. This avoids large gradients and overshooting the optimization goals. Adam will activate weight decay and also go through a warm-up phase.
