<a name="br1"></a>English - French Translator with RNN

1\. Synopsis

● English - French Translator with RNN is the capstone project in my AIND.

● My translator can archive 93% accuracy on word level from custom dataset that contains

○ 227 unique words from 1,823,250 words in English
○ 355 unique words from 1,961,295 words in French

● Implementing the translator, I use word embeding and encoding-decoding layer to improve model preformance.
● Following are libraries I use in this project:

○ Tensorflow: Deep Learning library from Google
○ Keras: A API for different libraries to build Deep Learning model
○ Numpy: A library for mathematics operations

2\. Repository Structure

● machine\_trnaslation.html: My machine translator in HTML
● machine\_trnaslation.ipynb: My machine translator in Jupyter Notebook
● helper.py: File reader for machine\_trnaslation.ipynb
● project\_tests.py: Unit test file




<a name="br2"></a>Image credit: [xiandong79.github.io](https://xiandong79.github.io/seq2seq-%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86)

**Goal:**

In this project,

● We build a deep neural network that functions as part of a machine translation pipeline.
● The pipeline accepts English text as input and returns the French translation.
● The goal is to achieve the highest translation accuracy possible.

**Background:**

● The ability to communicate with one another is a fundamental part of being human.
● There are nearly 7,000 different languages worldwide.

● As our world becomes increasingly connected, language translation provides a critical cultural and economic
 bridge between people from different countries and ethnic groups.

● Some of the more obvious use-cases include:

○ business: international trade, investment, contracts, finance

○ commerce: travel, purchase of foreign goods and services, customer support




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

● To translate a corpus of English text to French, we need to build a recurrent neural network (RNN).

● Before diving into the implementation, let's first build some intuition of RNNs and why they're useful for NLP tasks.

**RNN Overview**

● RNNs are designed to take sequences of text as inputs or return sequences of text as outputs, or both.
● They're called recurrent because the network's hidden layers have a loop in which the output from one time step
 becomes an input at the next time step.

● This recurrence serves as a form of memory. It allows contextual information to flow through the network so that

relevant outputs from previous time steps can be applied to network operations at the current time step.

● This is analogous to how we read. As you read this post, you're storing important pieces of information from

previous words and sentences and using it as context to understand each new word and sentence.

● Other types of neural networks can't do this. Imagine you're using a convolutional neural network (CNN) to perform

object detection in a movie.

● Currently, there's no way for information from objects detected in previous scenes to inform the model's detection

of objects in the current scene.




<a name="br5"></a>● For example, if a courtroom and judge were detected in a previous scene, that information could help correctly

classify the judge's gavel in the current scene (instead of misclassifying it as a hammer or mallet).

● But CNNs don't allow this type of time-series context to flow through the network like RNNs do.

**RNN Setup:**

● Depending on the use-case, you'll want to setup your RNN to handle inputs and outputs differently.

● For this project, we'll use a many-to-many process where the input is a sequence of English words and the output

is a sequence of French words (see fourth model from the left in the diagram below).

● Each rectangle is a vector and arrows represent functions (e.g. matrix multiply). Input vectors are in red, output
 vectors are in blue and green vectors hold the RNN's state (more on this soon).
● From left to right:

○ (1) Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image

classification).

○ (2) Sequence output (e.g. image captioning takes an image and outputs a sentence of words).

○ (3) Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or

negative sentiment).

○ (4) Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English
 and then outputs a sentence in French).

○ (5) Synced sequence input and output (e.g. video classification where we wish to label each frame of the

video).

● Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent

transformation (green) is fixed and can be applied as many times as we like.

Image and quote source: [karpathy.github.io](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)




<a name="br6"></a>**Building the Pipeline:**

Below is a summary of the various preprocessing and modeling steps. The high-level steps include:

1\. Preprocessing: load and examine data, cleaning, tokenization, padding

2\. Modeling: build, train, and test the model

3\. Prediction: generate specific translations of English to French, and compare the output translations to the ground

truth translations

4\. Iteration: iterate on the model, experimenting with different architectures

For a more detailed walkthrough including the source code, check out the Jupyter notebook in the main directory (// Link

).

**Toolset:**

● We use Keras for the frontend and TensorFlow for the backend in this project.

● I prefer using Keras on top of TensorFlow because the syntax is simpler, which makes building the model layers

more intuitive.

● However, there is a trade-off with Keras as you lose the ability to do fine-grained customizations.
● But this won't affect the models we're building in this project.

**Preprocessing:**

**Load & Examine Data**

● Here is a sample of the data. The inputs are sentences in English; the outputs are the corresponding translations

in French.




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

● For this project, each word and punctuation mark will be given a unique ID. (For other NLP projects, it might make
 sense to assign each character a unique ID.)

● When we run the tokenizer, it creates a word index, which is then used to convert each sentence to a vector.

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

**Modeling:**

First, let's breakdown the architecture of a RNN at a high level.

Referring to the diagram above, there are a few parts of the model we to be aware of:

1\. Inputs — Input sequences are fed into the model with one word for every time step. Each word is encoded as a
 unique integer or one-hot encoded vector that maps to the English dataset vocabulary.

2\. Embedding Layers — Embeddings are used to convert each word to a vector. The size of the vector depends on
 the complexity of the vocabulary.

3\. Recurrent Layers (Encoder) — This is where the context from word vectors in previous time steps is applied to the

current word vector.

4\. Dense Layers (Decoder) — These are typical fully connected layers used to decode the encoded input into the
 correct translation sequence.

5\. Outputs — The outputs are returned as a sequence of integers or one-hot encoded vectors which can then be

mapped to the French dataset vocabulary.




<a name="br12"></a>**Embeddings:**

● Embeddings allow us to capture more precise syntactic and semantic word relationships.
● This is achieved by projecting each word into n-dimensional space.

● Words with similar meanings occupy similar regions of this space; the closer two words are, the more similar they

are.

● And often the vectors between words represent useful relationships, such as gender, verb tense, or even

geopolitical relationships.

● Training embeddings on a large dataset from scratch requires a huge amount of data and computation.
● So, instead of doing it ourselves, we'd normally use a pre-trained embeddings package such as [GloVe](https://nlp.stanford.edu/projects/glove/)[ ](https://nlp.stanford.edu/projects/glove/)or

[word2vec](https://mubaris.com/2017/12/14/word2vec/).

● When used this way, embeddings are a form of transfer learning.

● However, since our dataset for this project has a small vocabulary and little syntactic variation, we'll use Keras to

train the embeddings ourselves.

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

**Bidirectional Layer:**

● Now that we understand how context flows through the network via the hidden state, let's take it a step further by

allowing that context to flow in both directions. This is what a bidirectional layer does.




<a name="br15"></a>● In the example above, the encoder only has historical context.

● But, providing future context can result in better model performance.

● This may seem counterintuitive to the way humans process language, since we only read in one direction.
● However, humans often require future context to interpret what is being said.

● In other words, sometimes we don't understand a sentence until an important word or phrase is provided at the

end. Happens this does whenever Yoda speaks.

● To implement this, we train two RNN layers simultaneously. The first layer is fed the input sequence as-is and the

second is fed a reversed copy.




<a name="br16"></a>**Hidden Layer — Gated Recurrent Unit (GRU):**

● Now let's make our RNN a little bit smarter.

● Instead of allowing all of the information from the hidden state to flow through the network, what if we could be

more selective?

● Perhaps some of the information is more relevant, while other information should be discarded.
● This is essentially what a gated recurrent unit (GRU) does.

● There are two gates in a GRU: an update gate and reset gate.

● [This](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)[ ](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)[article](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)[ ](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)by Simeon Kostadinov, explains these in detail. To summarize, the update gate (z) helps the model
 determine how much information from previous time steps needs to be passed along to the future.
● Meanwhile, the reset gate (r) decides how much of the past information to forget.




<a name="br17"></a>Image Credit: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/gru/)

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

a. Embedding Language Model (ELMo) — One of the biggest advances in [universal](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)[ ](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)[embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)[ ](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)in 2018 was

ELMo, developed by the [Allen](https://allennlp.org/)[ ](https://allennlp.org/)[Institute](https://allennlp.org/)[ ](https://allennlp.org/)[for](https://allennlp.org/)[ ](https://allennlp.org/)[AI](https://allennlp.org/). One of the major advantages of ELMo is that it addresses the



<a name="br19"></a>problem of polysemy, in which a single word has multiple meanings. ELMo is context-based (not
word-based), so different meanings for a word occupy different vectors within the embedding space. With
GloVe and word2vec, each word has only one representation in the embedding space. For example, the
word "queen" could refer to the matriarch of a royal family, a bee, a chess piece, or the 1970s rock band.
With traditional embeddings, all of these meanings are tied to a single vector for the word queen. With
ELMO, these are four distinct vectors, each with a unique set of context words occupying the same region
of the embedding space. For example, we'd expect to see words like queen, rook, and pawn in a similar
vector space related to the game of chess. And we'd expect to see queen, hive, and honey in a different
vector space related to bees. This provides a significant boost in semantic encoding.

b. Bidirectional Encoder Representations from [Transformers](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)[ ](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)(BERT). So far in 2019, the biggest

advancement in bidirectional embeddings has been [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html), which was open-sourced by Google. How is
BERT different?

Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the
vocabulary. For example, the word “bank” would have the same context-free representation in “bank account” and “bank
of the river.” Contextual models instead generate a representation of each word that is based on the other words in the
sentence. For example, in the sentence “I accessed the bank account,” a unidirectional contextual model would represent
“bank” based on “I accessed the” but not “account.” However, BERT represents “bank” using both its previous and next
context — “I accessed the ... account” — starting from the very bottom of a deep neural network, making it deeply
bidirectional. —Jacob Devlin and Ming-Wei Chang, [Google](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)[ ](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)[AI](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)[ ](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)[Blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
