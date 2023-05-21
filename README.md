

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

## Cleaning
No additional cleaning needs to be done at this point. The data has already been converted to lowercase and split so that there are spaces between all words and punctuation.

Note: For other NLP projects, you may need to perform additional steps such as removing HTML tags, stop words, punctuation, converting to tag representations, labeling parts of speech, or performing entity extraction.

## Tokenization
Next, we need to tokenize the data, which means converting the text to numerical values. This allows the neural network to perform operations on the input data. For this project, each word and punctuation mark will be given a unique ID. (In other NLP projects, it might make sense to assign each character a unique ID.) The tokenizer creates a word index, which is used to convert each sentence to a vector.

## Padding
When feeding our sequences of word IDs into the model, each sequence needs to be the same length. To achieve this, padding is added to any sequence that is shorter than the maximum length (i.e., shorter than the longest sentence).

## One-Hot Encoding (not used)
In this project, our input sequences are vectors containing a series of integers, where each integer represents an English word. We do not use one-hot encoding (OHE) in this project, but you may come across references to it in certain diagrams. OHE can be useful for representing categorical data where there is no ordinal relationship between different values. However, it can lead to long and sparse vectors. Since we are using embeddings in this project to further encode the word representations, we don't need to use OHE on our small dataset.

## Modeling
First, let's break down the architecture of an RNN at a high level.

Referring to the diagram above, there are a few parts of the model we need to be aware of:
- Inputs: Input sequences are fed into the model, with one word for every time step. Each word is encoded as a unique integer or one-hot encoded vector that maps to the English dataset vocabulary.
- Embedding Layers: Embeddings are used to convert each word to a vector representation. The size of the vector depends on the complexity of the vocabulary.
- Recurrent Layers (Encoder): This is where the context from word vectors in previous time steps is applied to the current word vector.
- Dense Layers (Decoder): These are typical fully connected layers used to decode the encoded input into the correct translation sequence.
- Outputs: The outputs are returned as a sequence of integers or one-hot encoded vectors, which can then be mapped to the French dataset vocabulary.

## Embeddings
Embeddings allow us to capture more precise syntactic and semantic word relationships by projecting each word into an n-dimensional space. Words with similar meanings occupy similar regions of this space, and the vectors between words represent useful relationships. Training embeddings on a large dataset from scratch requires significant data and computation. However, since our dataset for this project has a small vocabulary and little syntactic variation, we'll train the embeddings ourselves using Keras.

## Encoder & Decoder
Our sequence-to-sequence model consists of an encoder and decoder, which are two recurrent networks linked together. The encoder summarizes the input into a context variable (state), and the decoder uses this context to generate the output sequence. Both the encoder and decoder have loops that process each part of the sequence at different time steps. In the encoder, the input word is transformed with the hidden state, and the hidden state is passed to the next time step. In the decoder, the input at each time step is the previous word from the output sequence. The hidden state represents the relevant context flowing through the network.


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

## Bidirectional Layer

Now that we understand how context flows through the network via the hidden state, let's take it a step further by allowing that context to flow in both directions. This is what a bidirectional layer does.

In the example above, the encoder only has historical context. But providing future context can result in better model performance. This may seem counterintuitive to the way humans process language since we only read in one direction. However, humans often require future context to interpret what is being said. In other words, sometimes we don't understand a sentence until an important word or phrase is provided at the end. This happens whenever Yoda speaks. 😑 🙏

To implement this, we train two RNN layers simultaneously. The first layer is fed the input sequence as-is, and the second is fed a reversed copy.

## Hidden Layer — Gated Recurrent Unit (GRU)

Now let's make our RNN a little bit smarter. Instead of allowing all of the information from the hidden state to flow through the network, what if we could be more selective? Perhaps some of the information is more relevant, while other information should be discarded. This is essentially what a gated recurrent unit (GRU) does.

There are two gates in a GRU: an update gate and a reset gate. This article by Simeon Kostadinov explains these in detail. To summarize, the update gate (z) helps the model determine how much information from previous time steps needs to be passed along to the future. Meanwhile, the reset gate (r) decides how much of the past information to forget.

Image Credit: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2020/08/building-bidirectional-rnn-model-python/)

## Final Model

Now that we've discussed the various parts of our model, let's take a look at the code. Again, all of the source code is available [here](// Our Code Link).


def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
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
"""
    # Build the layers
    model = Sequential()
    # Embedding
    model.add(Embedding(english_vocab_size, 128, input_length=input_shape[1], input_shape=input_shape[1:]))
    # Encoder
    model.add(Bidirectional(GRU(128)))
    model.add(RepeatVector(output_sequence_length))
    # Decoder
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

##### &nbsp;

## Future Improvements

If I were to expand on it in the future, here's where I'd start:

1. **Do proper data split (training, validation, test):** Currently, there is no test set, only training and validation. To follow best practices, it's important to have a separate test set for evaluating the model's performance.

2. **LSTM + attention:** LSTM with attention has been a popular architecture for RNNs in recent years. It allows the model to focus on specific parts of the input sequence, enhancing its ability to capture important information. Incorporating LSTM with attention could further improve the model's performance.

3. **Train on a larger and more diverse text corpus:** The current text corpus used for training the model is small and lacks variability in syntax. To create a model that generalizes better and performs well on different types of text data, training on a larger dataset with more diverse grammar and sentence structures would be beneficial.

4. **Residual layers:** Adding residual layers to a deep LSTM RNN can help alleviate the vanishing gradient problem and improve the flow of information across layers. Residual connections have shown promising results in deep learning architectures.

5. **Embeddings:** When training on a larger dataset, utilizing pre-trained word embeddings such as word2vec or GloVe can enhance the model's performance. These embeddings capture rich semantic information and can provide a better representation of words.

6. **Embedding Language Model (ELMo):** ELMo, developed by the Allen Institute for AI, offers significant advancements in universal embeddings. It addresses the problem of polysemy, where a single word can have multiple meanings. ELMo provides context-based embeddings, allowing different meanings of a word to occupy distinct vectors within the embedding space. Incorporating ELMo could boost the semantic encoding of the model.

7. **Bidirectional Encoder Representations from Transformers (BERT):** BERT, developed by Google, has been a major breakthrough in bidirectional embeddings. Unlike context-free models like word2vec or GloVe, BERT generates word representations based on both previous and next context in a sentence. This bidirectional approach enables a deeper understanding of the word's meaning within the sentence. Exploring BERT-based models could further enhance the model's performance.

These improvements can contribute to creating a more robust and accurate model for language processing tasks.

