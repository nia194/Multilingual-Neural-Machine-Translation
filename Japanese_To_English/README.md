<a name="br1"></a>Japanese - English Translator with RNN

1\. Synopsis

● Japanese - English Translator with Trabsfomer.

● My translator can archive 93% accuracy on word level from custom dataset that contains

○ 62487 words in English
○ 62487 words in Japanese

● Implementing the translator, I use word embeding and encoding-decoding layer to improve model preformance.
● Following are libraries I use in this project:

○ Tensorflow: Deep Learning library from Google
○ Keras: A API for different libraries to build Deep Learning model
○ Numpy: A library for mathematics operations


2\. Repository Structure

● Jpn_to_Eng.ipynb: My machine translator in Jupyter Notebook
● dataset/eng_jpn.txt: a collection of text written in Japanese and translated into English
● dataset/kyoto_lexicon.csv: CSV file containing English-Japanese translations or descriptions of terms associated with Kyoto.




<a name="br2"></a>Image credit: [xiandong79.github.io](https://xiandong79.github.io/seq2seq-%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86)

**Goal:**

In this project,

● We build a deep neural network that functions as part of a machine translation pipeline.
● The pipeline accepts Japanese text as input and returns the English  translation.
● The goal is to achieve the highest translation accuracy possible.

**Background:**

● The ability to communicate with one another is a fundamental part of being human.
● There are nearly 7,000 different languages worldwide.

● In our interconnected world, language translation plays a vital role in fostering cultural and economic connections between individuals from diverse nations and ethnicities.

● Some evident applications of language translation include business activities such as international trade, investments, contracts, and finance, as well as commerce-related activities like travel, buying foreign products and services, and customer support.


<a name="br3"></a>○ media: accessing information via search, sharing information via social networks, localization of content

and advertising

○ education: sharing of ideas, collaboration, translation of research papers
○ government: foreign relations, negotiation

To meet this need:

● Technology companies are investing heavily in machine translation.

● This investment paired with recent advancements in deep learning have yielded major improvements in translation

quality.

● According to Google, [switching](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[to](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[deep](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[learning](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[produced](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[a](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[60%](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[increase](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[in](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[translation](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[accuracy](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)[ ](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate)compared to the
 phrase-based approach used previously.

● Today, translation applications from Google and Microsoft can translate over 100 different languages and are

approaching human-level accuracy for many of them.

However, while machine translation has made lots of progress, it's still not perfect.

Bad translation or extreme carnivorism?




<a name="br4"></a>**Approach:**

● To translate a corpus of Japanese text to English, we need to use Transformer Model.

**Preprocessing:**

**Load & Examine Data**

● Here is a sample of the data. The inputs are sentences in japanese; the outputs are the corresponding translations

in English.




<a name="br7"></a>● When we run a word count, we can see that the vocabulary for the dataset is quite small. This was by design for

this project. This allows us to train the models in a reasonable time.

**Cleaning**

● No additional cleaning needs to be done at this point.

● The data has already been converted to lowercase and split so that there are spaces between all words and

punctuation.

● Note: For other NLP projects you may need to perform additional steps such as: remove HTML tags, remove stop

words, remove punctuation or convert to tag representations, label the parts of speech, or perform entity
extraction.




<a name="br8"></a>**Tokenization:**

● Next we need to tokenize the data—i.e., convert the text to numerical values.
● This allows the neural network to perform operations on the input data.

● We are using spacy library for tokenization of english and japanese language.

**Padding:**

● When we feed our sequences of word IDs into the model, each sequence needs to be the same length. To achieve

this, padding is added to any sequence that is shorter than the max length (i.e. shorter than the longest sentence).

**One-Hot Encoding (not used):**

● In this project, our input sequences will be a vector containing a series of integers.
● Each integer represents an English word (as seen above).




<a name="br9"></a>● However, in other projects, sometimes an additional step is performed to convert each integer into a one-hot

encoded vector.

● We don't use one-hot encoding (OHE) in this project, but you'll see references to it in certain diagrams (like the

one below). I just didn't want you to get confused.





<a name="br10"></a>


<a name="br11"></a>● One of the advantages of OHE is efficiency since it can [run](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[ ](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[at](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[ ](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[a](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[ ](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[faster](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[ ](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[clock](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[ ](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[rate](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[ ](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[than](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[ ](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[other](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[ ](https://en.wikipedia.org/wiki/One-hot#cite_note-2)[encodings](https://en.wikipedia.org/wiki/One-hot#cite_note-2).

● The other advantage is that OHE better represents categorical data where there is no ordinal relationship between

different values.

● For example, let's say we're classifying animals as either a mammal, reptile, fish, or bird.

● If we encode them as 1, 2, 3, 4 respectively, our model may assume there is a natural ordering between them,

which there isn't.

● It's not useful to structure our data such that mammal comes before reptile and so forth.
● This can mislead our model and cause poor results.

● However, if we then apply one-hot encoding to these integers, changing them to binary representations—1000,

0100, 0010, 0001 respectively—then no ordinal relationship can be inferred by the model.

But, one of the drawbacks of OHE is that the vectors can get very long and sparse.

● The length of the vector is determined by the vocabulary, i.e. the number of unique words in your text corpus.
● As we saw in the data examination step above, our vocabulary for this project is very small—only 227 English
 words and 355 French words.

● By comparison, the [Oxford](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[ ](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[English](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[ ](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[Dictionary](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[ ](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[has](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[ ](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[172,000](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[ ](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/)[words](https://en.oxforddictionaries.com/explore/how-many-words-are-there-in-the-english-language/).

● But, if we include various proper nouns, words tenses, and slang there could be millions of words in each

language.

● For example, [Google's](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/)[ ](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/)[word2vec](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/)[ ](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/)is trained on a vocabulary of 3 million unique words. If we used OHE on this

vocabulary, the vector for each word would include one positive value (1) surrounded by 2,999,999 zeros!

And, since we're using embeddings (in the next step) to further encode the word representations, we don't need to bother
with OHE. Any efficiency gains aren't worth it on a data set this small.


**Encoder & Decoder:**

● Our sequence-to-sequence model links two recurrent networks: an encoder and decoder.
● The encoder summarizes the input into a context variable, also called the state.
● This context is then decoded and the output sequence is generated.




<a name="br13"></a>Image credit: [Udacity](https://classroom.udacity.com/nanodegrees/nd101/parts/4f636f4e-f9e8-4d52-931f-a49a0c26b710/modules/c1558ffb-9afd-48fa-bf12-b8f29dcb18b0/lessons/43ccf91e-7055-4833-8acc-0e2cf77696e8/concepts/be468484-4bd5-4fb0-82d6-5f5697af07da)

● Since both the encoder and decoder are recurrent, they have loops which process each part of the sequence at

different time steps.

● To picture this, it's best to unroll the network so we can see what's happening at each time step.
● In the example below, it takes four time steps to encode the entire input sequence. At each time step, the encoder
 "reads" the input word and performs a transformation on its hidden state.

● Then it passes that hidden state to the next time step.

● Keep in mind that the hidden state represents the relevant context flowing through the network.

● The bigger the hidden state, the greater the learning capacity of the model, but also the greater the computation

requirements.

● We'll talk more about the transformations within the hidden state when we cover gated recurrent units (GRU).




<a name="br14"></a>Image credit: modified version from [Udacity](https://classroom.udacity.com/nanodegrees/nd101/parts/4f636f4e-f9e8-4d52-931f-a49a0c26b710/modules/c1558ffb-9afd-48fa-bf12-b8f29dcb18b0/lessons/43ccf91e-7055-4833-8acc-0e2cf77696e8/concepts/f999d8f6-b4c1-4cd0-811e-4767b127ae50)

● For now, notice that for each time step after the first word in the sequence there are two inputs:
 ○ the hidden state and

○ a word from the sequence.

● For the encoder, it's the next word in the input sequence.

● For the decoder, it's the previous word from the output sequence.

● Also, remember that when we refer to a "word," we really mean the vector representation of the word which comes

from the embedding layer.

\
**Final Model:**

● Now that we've discussed the various parts of our model, let's take a look at the code. Again, all of the source
 code is available (// Our Code Link).

def model\_final (input\_shape, output\_sequence\_length, english\_vocab\_size, french\_vocab\_size):

"""

Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN
:param input\_shape: Tuple of input shape

:param output\_sequence\_length: Length of output sequence
:param english\_vocab\_size: Number of unique English words in the dataset
:param french\_vocab\_size: Number of unique French words in the dataset
:return: Keras model built, but not trained

"""

\# Hyperparameters learning\_rate = 0.003

\# Build the layers




<a name="br18"></a>model = Sequential() # Embedding

model.add(Embedding(english\_vocab\_size, 128, input\_length=input\_shape[1],
 input\_shape=input\_shape[1:]))

\# Encoder

model.add(Bidirectional(GRU(128)))
model.add(RepeatVector(output\_sequence\_length))
# Decoder

model.add(Bidirectional(GRU(128, return\_sequences=True)))
model.add(TimeDistributed(Dense(512, activation='relu')))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(french\_vocab\_size, activation='softmax')))
model.compile(loss=sparse\_categorical\_crossentropy,
 optimizer=Adam(learning\_rate),

metrics=['accuracy']) return model

**Results:**

The results from the final model can be found in cell 20 of the (// Our code Link).
Validation accuracy: 97.5% Training time: 23 epochs

**Future Improvements:**

If I were to expand on it in the future, here's where I'd start.

1\. Do proper data split (training, validation, test) — Currently there is no test set, only training and validation.
 Obviously this doesn't follow best practices.

2\. LSTM + attention — This has been the de facto architecture for RNNs over the past few years, although there are
[ ](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0)[some](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0)[ ](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0)[limitations](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0). I didn't use LSTM because I'd already implemented it in TensorFlow in another project (found
[ ](https://github.com/tommytracey/udacity/tree/master/deep-learning-nano/projects/4-language-translation#build-the-neural-network)[here](https://github.com/tommytracey/udacity/tree/master/deep-learning-nano/projects/4-language-translation#build-the-neural-network)), and I wanted to experiment with GRU + Keras for this project.

3\. Train on a larger and more diverse text corpus — The text corpus and vocabulary for this project are quite small
 with little variation in syntax. As a result, the model is very brittle. To create a model that generalizes better, you'll
 need to train on a larger dataset with more variability in grammar and sentence structure.

4\. Residual layers — Yo u could add residual layers to a deep LSTM RNN, as described in [this](https://arxiv.org/abs/1701.03360)[ ](https://arxiv.org/abs/1701.03360)[paper](https://arxiv.org/abs/1701.03360). Or, use
 residual layers as an alternative to LSTM and GRU, as described [here](http://www.mdpi.com/2078-2489/9/3/56/pdf).

5\. Embeddings — If you're training on a larger dataset, you should definitely use a pre-trained set of embeddings

such as [word2vec](https://mubaris.com/2017/12/14/word2vec/)[ ](https://mubaris.com/2017/12/14/word2vec/)or [GloVe](https://nlp.stanford.edu/projects/glove/). Even better, use ELMo or BERT.
