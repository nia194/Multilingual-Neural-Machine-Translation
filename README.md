

# English - French Translator with RNN

## Synopsis
English - French Translator with RNN is the capstone project in my AIND.

My translator can achieve 93% accuracy on the word level using a custom dataset that contains:
- 227 unique words from 1,823,250 words in English
- 355 unique words from 1,961,295 words in French

To implement the translator, I use word embedding and encoding-decoding layers to improve model performance.

The following libraries were used in this project:
- Tensorflow: Deep Learning library from Google
- Keras: An API for different libraries to build Deep Learning models
- Numpy: A library for mathematical operations

## Repository Structure
- `machine_translation.html`: My machine translator in HTML
- `machine_translation.ipynb`: My machine translator in Jupyter Notebook
- `helper.py`: File reader for machine_translation.ipynb
- `project_tests.py`: Unit test file

# Machine Translation with Deep Neural Network

## Goal
In this project, we build a deep neural network that functions as part of a machine translation pipeline. The pipeline accepts English text as input and returns the French translation. The goal is to achieve the highest translation accuracy possible.

## Background
The ability to communicate with one another is a fundamental part of being human. There are nearly 7,000 different languages worldwide. As our world becomes increasingly connected, language translation provides a critical cultural and economic bridge between people from different countries and ethnic groups. Some of the more obvious use-cases include:

- Business: international trade, investment, contracts, finance
- Commerce: travel, purchase of foreign goods and services, customer support
- Media: accessing information via search, sharing information via social networks, localization of content and advertising
- Education: sharing of ideas, collaboration, translation of research papers
- Government: foreign relations, negotiation

## Meeting the Need
Technology companies are investing heavily in machine translation. This investment, paired with recent advancements in deep learning, has yielded major improvements in translation quality. According to Google, switching to deep learning produced a 60% increase in translation accuracy compared to the phrase-based approach used previously. Today, translation applications from Google and Microsoft can translate over 100 different languages and are approaching human-level accuracy for many of them.

However, while machine translation has made lots of progress, it's still not perfect. 😬


<img src="images/.png" width="50%" align="top-left" alt="" title="RNN" />

_Bad translation or extreme carnivorism?_


##### &nbsp;

# Machine Translation: Bad Translation or Extreme Carnivorism?

## Approach
To translate a corpus of English text to French, we need to build a recurrent neural network (RNN). Before diving into the implementation, let's first build some intuition of RNNs and why they're useful for NLP tasks.

### RNN Overview
RNNs are designed to take sequences of text as inputs or return sequences of text as outputs, or both. They're called recurrent because the network's hidden layers have a loop in which the output from one time step becomes an input at the next time step. This recurrence serves as a form of memory, allowing contextual information to flow through the network and apply relevant outputs from previous time steps to the current time step.

This is analogous to how we read, where we store important information from previous words and sentences as context to understand each new word and sentence. Other types of neural networks, such as convolutional neural networks (CNNs), don't allow this type of time-series context to flow through the network like RNNs.

### RNN Setup
Depending on the use-case, you'll want to set up your RNN to handle inputs and outputs differently. For this project, we'll use a many-to-many process where the input is a sequence of English words and the output is a sequence of French words.

### Building the Pipeline
Below is a summary of the various preprocessing and modeling steps, including preprocessing, modeling, prediction, and iteration. For a more detailed walkthrough and source code, check out the Jupyter notebook in the main directory.

### Toolset
We use Keras for the frontend and TensorFlow for the backend in this project. Keras simplifies the syntax and makes building model layers more intuitive, although it may limit fine-grained customizations. However, this won't affect the models we're building in this project.

### Preprocessing
#### Load & Examine Data
The data consists of English sentences as inputs and their corresponding French translations as outputs. The vocabulary size is intentionally kept small for faster model training.

#### Cleaning
No additional cleaning is required at this point. The data has already been converted to lowercase and split with spaces between words and punctuation.

Note: For other NLP projects, additional steps such as removing HTML tags, stop words, or punctuation, or performing entity extraction may be necessary.

And this...

_Note:_ For other NLP projects you may need to perform additional steps such as: remove HTML tags, remove stop words, remove punctuation or convert to tag representations, label the parts of speech, or perform entity extraction.  

### Tokenization
Next we need to tokenize the data&mdash;i.e., convert the text to numerical values. This allows the neural network to perform operations on the input data. For this project, each word and punctuation mark will be given a unique ID. (For other NLP projects, it might make sense to assign each character a unique ID.)

When we run the tokenizer, it creates a word index, which is then used to convert each sentence to a vector.

> <img src="images/tok.png" width="100%" align="top-left" alt="" title="Tokenizer output" />

### Padding
When we feed our sequences of word IDs into the model, each sequence needs to be the same length. To achieve this, padding is added to any sequence that is shorter than the max length (i.e. shorter than the longest sentence).

> <img src="images/padding.png" width="50%" align="top-left" alt="" title="Tokenizer output" />

### One-Hot Encoding (not used)
In this project, our input sequences will be a vector containing a series of integers. Each integer represents an English word (as seen above). However, in other projects, sometimes an additional step is performed to convert each integer into a one-hot encoded vector. We don't use one-hot encoding (OHE) in this project, but you'll see references to it in certain diagrams (like the one below). I just didn't want you to get confused.  

<img src="images/RNN-architecture.png" width="40%" align="right" alt="" title="RNN architecture" />

One of the advantages of OHE is efficiency since it can [run at a faster clock rate than other encodings](https://en.wikipedia.org/wiki/One-hot#cite_note-2). The other advantage is that OHE better represents categorical data where there is no ordinal relationship between different values. For example, let's say we're classifying animals as either a mammal, reptile, fish, or bird. If we encode them as 1, 2, 3, 4 respectively, our model may assume there is a natural ordering between them, which there isn't. It's not useful to structure our data such that mammal comes before reptile and so forth. This can mislead our model and cause poor results. However, if we then apply one-hot encoding to these integers, changing them to binary representations&mdash;1000, 0100, 0010, 0001 respectively&mdash;then no ordinal relationship can be inferred by the model.

But, one of the drawbacks of OHE is that the vectors can get very long and sparse. The length of the vector is determined by the vocabulary, i.e. the number of unique words in your text corpus. As we saw in the data examination step above, our vocabulary for this project is very small&mdash;only 227 English words and 355 French words. By comparison, the [Oxford English Dictionary has 172,000 words](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/). But, if we include various proper nouns, words tenses, and slang there could be millions of words in each language. For example, [Google's word2vec](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/) is trained on a vocabulary of 3 million unique words. If we used OHE on this vocabulary, the vector for each word would include one positive value (1) surrounded by 2,999,999 zeros!

And, since we're using embeddings (in the next step) to further encode the word representations, we don't need to bother with OHE. Any efficiency gains aren't worth it on a data set this small.  


##### &nbsp;

## Modeling
First, let's breakdown the architecture of a RNN at a high level. Referring to the diagram above, there are a few parts of the model we to be aware of:

1. **Inputs** &mdash; Input sequences are fed into the model with one word for every time step. Each word is encoded as a unique integer or one-hot encoded vector that maps to the English dataset vocabulary.
1. **Embedding Layers** &mdash; Embeddings are used to convert each word to a vector. The size of the vector depends on the complexity of the vocabulary.
1. **Recurrent Layers (Encoder)** &mdash; This is where the context from word vectors in previous time steps is applied to the current word vector.
1. **Dense Layers (Decoder)** &mdash; These are typical fully connected layers used to decode the encoded input into the correct translation sequence.
1. **Outputs** &mdash; The outputs are returned as a sequence of integers or one-hot encoded vectors which can then be mapped to the French dataset vocabulary.

##### &nbsp;

### Embeddings
Embeddings allow us to capture more precise syntactic and semantic word relationships. This is achieved by projecting each word into n-dimensional space. Words with similar meanings occupy similar regions of this space; the closer two words are, the more similar they are. And often the vectors between words represent useful relationships, such as gender, verb tense, or even geopolitical relationships.

<img src="images/embedding-words.png" width="100%" align-center="true" alt="" title="Gated Recurrent Unit (GRU)" />

Training embeddings on a large dataset from scratch requires a huge amount of data and computation. So, instead of doing it ourselves, we'd normally use a pre-trained embeddings package such as [GloVe](https://nlp.stanford.edu/projects/glove/) or [word2vec](https://mubaris.com/2017/12/14/word2vec/). When used this way, embeddings are a form of transfer learning. However, since our dataset for this project has a small vocabulary and little syntactic variation, we'll use Keras to train the embeddings ourselves.

##### &nbsp;

### Encoder & Decoder
Our sequence-to-sequence model links two recurrent networks: an encoder and decoder. The encoder summarizes the input into a context variable, also called the state. This context is then decoded and the output sequence is generated.

##### &nbsp;

<img src="images/encoder-decoder-context.png" width="60%" align="top-left" alt="" title="Encoder Decoder" />

_Image credit: [Udacity](https://classroom.udacity.com/nanodegrees/nd101/parts/4f636f4e-f9e8-4d52-931f-a49a0c26b710/modules/c1558ffb-9afd-48fa-bf12-b8f29dcb18b0/lessons/43ccf91e-7055-4833-8acc-0e2cf77696e8/concepts/be468484-4bd5-4fb0-82d6-5f5697af07da)_

##### &nbsp;

Since both the encoder and decoder are recurrent, they have loops which process each part of the sequence at different time steps. To picture this, it's best to unroll the network so we can see what's happening at each time step.

In the example below, it takes four time steps to encode the entire input sequence. At each time step, the encoder "reads" the input word and performs a transformation on its hidden state. Then it passes that hidden state to the next time step. Keep in mind that the hidden state represents the relevant context flowing through the network. The bigger the hidden state, the greater the learning capacity of the model, but also the greater the computation requirements. We'll talk more about the transformations within the hidden state when we cover gated recurrent units (GRU).

<img src="images/encoder-decoder-translation.png" width="100%" align="top-left" alt="" title="Encoder Decoder" />

_Image credit: modified version from [Udacity](https://classroom.udacity.com/nanodegrees/nd101/parts/4f636f4e-f9e8-4d52-931f-a49a0c26b710/modules/c1558ffb-9afd-48fa-bf12-b8f29dcb18b0/lessons/43ccf91e-7055-4833-8acc-0e2cf77696e8/concepts/f999d8f6-b4c1-4cd0-811e-4767b127ae50)_

##### &nbsp;

For now, notice that for each time step after the first word in the sequence there are two inputs: the hidden state and a word from the sequence. For the encoder, it's the _next_ word in the input sequence. For the decoder, it's the _previous_ word from the output sequence.

Also, remember that when we refer to a "word," we really mean the _vector representation_ of the word which comes from the embedding layer.

##### &nbsp;

### Bidirectional Layer
Now that we understand how context flows through the network via the hidden state, let's take it a step further by allowing that context to flow in both directions. This is what a bidirectional layer does.

<img src="images/yoda.jpg" width="40%" align="right" alt="" title="Yoda" />

In the example above, the encoder only has historical context. But, providing future context can result in better model performance. This may seem counterintuitive to the way humans process language, since we only read in one direction. However, humans often require future context to interpret what is being said. In other words, sometimes we don't understand a sentence until an important word or phrase is provided at the end. Happens this does whenever Yoda speaks. :expressionless: :pray:

To implement this, we train two RNN layers simultaneously. The first layer is fed the input sequence as-is and the second is fed a reversed copy.

<img src="images/bidirectional.png" width="70%" align="center" alt="" title="Bidirectional Layer" />

##### &nbsp;

### Hidden Layer &mdash; Gated Recurrent Unit (GRU)
Now let's make our RNN a little bit smarter. Instead of allowing _all_ of the information from the hidden state to flow through the network, what if we could be more selective? Perhaps some of the information is more relevant, while other information should be discarded. This is essentially what a gated recurrent unit (GRU) does.

There are two gates in a GRU: an update gate and reset gate. [This article](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be) by Simeon Kostadinov, explains these in detail. To summarize, the **update gate (z)** helps the model determine how much information from previous time steps needs to be passed along to the future. Meanwhile, the **reset gate (r)** decides how much of the past information to forget.

##### &nbsp;

<img src="images/gru.png" width="70%" align-center="true" alt="" title="Gated Recurrent Unit (GRU)" />

_Image Credit: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/gru/)_

##### &nbsp;

### Final Model
Now that we've discussed the various parts of our model, let's take a look at the code. Again, all of the source code is available [here in the notebook](machine_translation.ipynb).

```python

def  model_final (input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Hyperparameters
    learning_rate = 0.003

    # Build the layers    
    model = Sequential()
    # Embedding
    model.add(Embedding(english_vocab_size, 128, input_length=input_shape[1],
                         input_shape=input_shape[1:]))
    # Encoder
    model.add(Bidirectional(GRU(128)))
    model.add(RepeatVector(output_sequence_length))
    # Decoder
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
```
##### &nbsp;

## Results
The results from the final model can be found in cell 20 of the [notebook](machine_translation.ipynb).

Validation accuracy: 97.5%

Training time: 23 epochs


##### &nbsp;

## Future Improvements
If I were to expand on it in the future, here's where I'd start.

1. **Do proper data split (training, validation, test)** &mdash; Currently there is no test set, only training and validation. Obviously this doesn't follow best practices.
1. **LSTM + attention** &mdash; This has been the de facto architecture for RNNs over the past few years, although there are [some limitations](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0). I didn't use LSTM because I'd already implemented it in TensorFlow in another project (found [here](https://github.com/tommytracey/udacity/tree/master/deep-learning-nano/projects/4-language-translation#build-the-neural-network)), and I wanted to experiment with GRU + Keras for this project.
1. **Train on a larger and more diverse text corpus** &mdash; The text corpus and vocabulary for this project are quite small with little variation in syntax. As a result, the model is very brittle. To create a model that generalizes better, you'll need to train on a larger dataset with more variability in grammar and sentence structure.  
1. **Residual layers** &mdash; You could add residual layers to a deep LSTM RNN, as described in [this paper](https://arxiv.org/abs/1701.03360). Or, use residual layers as an alternative to LSTM and GRU, as described [here](http://www.mdpi.com/2078-2489/9/3/56/pdf).
1. **Embeddings** &mdash; If you're training on a larger dataset, you should definitely use a pre-trained set of embeddings such as [word2vec](https://mubaris.com/2017/12/14/word2vec/) or [GloVe](https://nlp.stanford.edu/projects/glove/). Even better, use ELMo or BERT.
 - **Embedding Language Model (ELMo)** &mdash; One of the biggest advances in [universal embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a) in 2018 was ELMo, developed by the [Allen Institute for AI](https://allennlp.org). One of the major advantages of ELMo is that it addresses the problem of polysemy, in which a single word has multiple meanings. ELMo is context-based (not word-based), so different meanings for a word occupy different vectors within the embedding space. With GloVe and word2vec, each word has only one representation in the embedding space. For example, the word "queen" could refer to the matriarch of a royal family, a bee, a chess piece, or the 1970s rock band. With traditional embeddings, all of these meanings are tied to a single vector for the word _queen_. With ELMO, these are four distinct vectors, each with a unique set of context words occupying the same region of the embedding space. For example, we'd expect to see words like _queen_, _rook_, and _pawn_ in a similar vector space related to the game of chess. And we'd expect to see _queen_, _hive_, and _honey_ in a different vector space related to bees. This provides a significant boost in semantic encoding.
 - **Bidirectional Encoder Representations from [Transformers](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) (BERT)**. So far in 2019, the biggest advancement in bidirectional embeddings has been [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html), which was open-sourced by Google. How is BERT different?
> Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary. For example, the word “bank” would have the same context-free representation in “bank account” and “bank of the river.” Contextual models instead generate a representation of each word that is based on the other words in the sentence. For example, in the sentence “I accessed the bank account,” a unidirectional contextual model would represent “bank” based on “I accessed the” but not “account.” However, BERT represents “bank” using both its previous and next context — “I accessed the ... account” — starting from the very bottom of a deep neural network, making it deeply bidirectional.
> &mdash;Jacob Devlin and Ming-Wei Chang, [Google AI Blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

##### &nbsp;

### Contact
I hope you found this useful. If you have any feedback, I’d love to hear it. Feel free to post in the comments.

If you’d like to inquire about collaboration or career opportunities you can find me [here on LinkedIn](https://www.linkedin.com/in/thomastracey/) or view [my portfolio here](https://ttracey.com/).

##### &nbsp;

---

# Project Starter Code
In case you want to run this project yourself, below is the project starter code.

## Setup
The original Udacity repo for this project can be found [here](https://github.com/udacity/aind2-nlp-capstone).

This project requires GPU acceleration to run efficiently. Support is available to use either of the following two methods for accessing GPU-enabled cloud computing resources.

### Udacity Workspaces (Recommended)

Udacity Workspaces provide remote connection to GPU-enabled instances right from the classroom. Refer to the classroom lesson for this project to find an overview of navigating & using Jupyter notebook Workspaces.

### Amazon Web Services (Optional)

Please refer to the Udacity instructions for setting up a GPU instance for this project, and refer to the project instructions in the classroom for setup. The recommended AMI should include compatible versions of all required software and libraries to complete the project. [link for AIND students](https://classroom.udacity.com/nanodegrees/nd889/parts/16cf5df5-73f0-4afa-93a9-de5974257236/modules/53b2a19e-4e29-4ae7-aaf2-33d195dbdeba/lessons/2df3b94c-4f09-476a-8397-e8841b147f84/project)

### Install
- Python 3
- NumPy
- TensorFlow 1.x
- Keras 2.x

## Submission
When you are ready to submit your project, do the following steps:
1. Ensure you pass all points on the [rubric](https://review.udacity.com/#!/rubrics/1004/view).
2. Submit the following in a zip file:
  - `helper.py`
  - `machine_translation.ipynb`
  - `machine_translation.html`

### Converting to HTML

There are several ways to generate an HTML copy of the notebook:

 - Running the last cell of the notebook will export an HTML copy

 - Navigating to **File -> Download as -> HTML (.html)** within the notebook

 - Using `nbconvert` from the command line

    $ pip install nbconvert
    $ nbconvert machine_translation.ipynb
