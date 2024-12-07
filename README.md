## Attention Is All You Need Paper Implementation ✨📝

This is my from-scratch implementation of the original transformer architecture from the following paper: [Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.](https://arxiv.org/abs/1706.03762)

My Code Implementation: [Kaggle](https://www.kaggle.com/code/amankshetri/attention-is-all-you-need-paper-transformer)

*If you wanna quickly try out the attention model from parallel text examples, then checkout this : [Kaggle](https://www.kaggle.com/code/amankshetri/easy-transformer-from-parallel-text-examples)*

*But I recommend you to implement the above main code.*

<a href=https://arxiv.org/pdf/1706.03762.pdf>
  <p align="center">
    <img width="540" height="700" src="/assets/banner_paper.jpg" alt="Attention Paper Banner Img">
  </p>
</a>


## Table of Contents

1. [Introduction](#introduction) 
2. [Pre-requisites](#prerequisites) 
3. [Architecture Overview](#architecture-overview) 
4. [Implementation Details](#implementation-details) 
5. [Training](#training) 
6. [Evaluation](#evaluation) 
7. [Use Cases](#usecases) 
    * [Dataset](#dataset) 
8. [Contributions](#contributions) 
9. [References](#references)


## 1. Introduction 📘

The `Transformer` model, proposed in the paper *"Attention Is All You Need"*, eliminates the need for recurrent architectures (RNNs) and instead uses a `self-attention mechanism` to process sequential data. This allows the model to better capture relationships within data and enables parallelization, significantly improving training efficiency.

In this repository, I implement the core ideas presented in the paper and provide a clear walkthrough of how to implement and train the Transformer for various NLP tasks.


## 2. Pre-requisites 🛠️

Before running the implementation, ensure you have the following dependencies:

- Python 3.x
- TensorFlow / PyTorch (depending on your preference)
- NumPy
- Matplotlib (for visualizations)
- scikit-learn (for model evaluation)
  
You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## 3. Architecture Overview 🏗️
The architecture of the Transformer model consists of two main parts: the `Encoder` and the `Decoder`. Both of these components use stacked layers of multi-head self-attention and position-wise feedforward networks. 

<p align="center">
  <img width="600" height="540" src="/assets/transformer_architecture.png" alt="Transformer Architecture Diagram">
</p>


### Key Components:
* `Self-Attention`: Helps the model focus on different parts of the input sequence when encoding/decoding.
* `Positional Encoding`: Adds information about the relative positions of words in the sequence.
* `Multi-Head Attention`: Multiple attention mechanisms run in parallel, allowing the model to learn from different aspects of the input data simultaneously.
* `Feedforward Networks`: Simple neural networks that process each token individually after the attention layer.


## 4. Implementation Details 🧩
The implementation is based on the architecture described in the paper and follows these key steps:


a. `Input Processing`:

* Tokenization of input text.
* Conversion of tokens to embeddings.
* Adding positional encoding to token embeddings.


b. `Encoder Layer`:

* Multi-Head Self-Attention.
* Add & Normalize.
* Position-Wise Feedforward Networks.

c. `Decoder Layer`:

* Multi-Head Self-Attention.
* Encoder-Decoder Attention.
* Position-Wise Feedforward Networks.

d. `Final Output`:

* Linear layer with softmax activation for generating the output sequence.

The entire model can be built using either TensorFlow or PyTorch. You can switch between frameworks by selecting the appropriate implementation.



## 5. Training 🏋️‍♂️
The Transformer model is trained using supervised learning on large-scale datasets (e.g., language translation). The training process involves:

* `Loss Function`: Categorical Cross-Entropy Loss.
* `Optimization`: Adam optimizer with learning rate scheduling.
* `Metrics`: Perplexity and BLEU score for language translation tasks.

<p align="center">
  <img src="/assets/training_img.png" alt="Model Training Image">
</p>


## 6. Evaluation 📊
After training, evaluate the model's performance on validation and test datasets. The evaluation script calculates metrics such as:

* `BLEU Score`: For machine translation tasks.
* `Perplexity`: For language modeling tasks.

<p align="center">
  <img src="/assets/testing_img.png" alt="Model Testing Image">
</p>


## 7. Use Cases 🚀

This section highlights various use cases for the **Attention Is All You Need** model, demonstrating its potential in practical applications. 

One of the key use cases for this model is `Language Translation`, where it can be trained to translate between different languages.


## Dataset 📂

For training and evaluating the model, we use the `English-French Language Translation Dataset` from Kaggle. This dataset provides parallel English-French sentences for machine translation tasks.

### Dataset Overview:

- **Source**: [Kaggle - Language Translation (English/French)](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench/)
- **Content**: The dataset consists of pairs of sentences in English and their corresponding French translations. This is a great resource for training models on language translation tasks.
- **Format**: The dataset is provided in `.csv` format, containing two columns: one for the English sentence and the other for the French translation.
- **Size**: The dataset contains approximately `10,000` pairs of English-French sentences, ideal for training a translation model.

<p align="center">
  <img src="/assets/dataset.png" alt="Dataset Category Image">
</p>


#### How to Use the Dataset:

1. **Download the Dataset**:
   - Go to the [Kaggle page](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench/) and download the dataset.
   - If you don't have a Kaggle account, you need to create one and accept the dataset's terms of use.

2. **Dataset Structure**:
   - The dataset contains the following columns:
     - `English`: The sentence in English.
     - `French`: The corresponding translation in French.


## 8. Contributions 🤝
Contributions are welcome! If you find any bugs, issues, or have suggestions for improvements, feel free to open an issue or submit a pull request.

To contribute:

1. Fork the repository.
2. Clone your fork and create a new branch for your changes.
3. Make your changes and ensure the code passes all tests.
4. Submit a pull request to the main repository.


## 9. References 📚

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. A., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. [Link](https://arxiv.org/abs/1706.03762)