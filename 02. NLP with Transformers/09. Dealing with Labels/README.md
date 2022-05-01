## **Transformers Efficient in Production**

The [**Transformers & Production**](https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/08.%20Transformers%20in%20Production/Transformers%26Production.ipynb) notebook contains information and implementation of Case Study: Intent Detection, Performance Benchmark, Computing Accuracy, Model Size, Knowledge Distillation, KL Divergence, Hyperparameters with Optuna, Model Quantization, Optimization with ONNX and ONNX Runtime.

**Note:**
  - 📝[**Transformers & Production**](https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/08.%20Transformers%20in%20Production/Transformers%26Production.ipynb)

**Knowledge Distillation**
- Knowledge distillation is a general purpose method for training a smaller student model to mimic the behavior of a slower, larger, but better performing teacher model. The KL divergence expects the inputs in the form of log probabilities and labels as normal probabilities. So, we have used log softmax to normalize the student's logits while teacher's logits are converted to probabilities with a standard softmax.
