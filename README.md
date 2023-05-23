# Multilingual Neural Machine Translation

Hi, How are you.  
Hallo, wie geht es dir.  
Ciao, come stai.  
Salut comment ça va.  
こんにちは、元気ですか。


## Description 

   In this project, we aim to develop a bi-directional translation system for multiple languages, including Dutch, Japanese, French, Italian, and English. Our focus is on building and comparing various neural network models such as Simple RNN, LSTM, Bi-directional LSTM, and Encoder-Decoder models. We will evaluate the performance of these models using metrics like Accuracy, BLEU Score, ROUGE Score, and F1-Score. Additionally, we will explore the use of transfer learning techniques, leveraging pretraining on large-scale multilingual data, to enhance translation quality. By conducting experiments and analyzing results, we aim to improve the accuracy and efficiency of bi-directional translation across multiple languages


## Getting Started
To get started with using these notebooks, you can do the following:

1. Clone this repository to your local machine or server.
2. Open the notebook in your preferred environment (e.g. Google Colab, Kaggle, or a good GPU computer).
3. Follow the instructions in the notebook to train and test the models.

# Usage
There are three notebooks available in this repository:

1. English-to-German-to-English: This notebook contains the implementation of neural machine translation.
2. English-to-Italian-to-English: This notebook contains the implementation of neural machine translation.
3. English-to-French-to-English: This notebook contains the implementation of neural machine translation.
4. English-to-Japanese-to-English: This notebook contains the implementation of neural machine translation.
To use any of the notebooks, open the corresponding notebook in your preferred environment and follow the instructions to train and test the models.

# Dataset
The dataset used in this project is available at Parallel Corpora (https://www.clarin.eu/resource-families/parallel-corpora) website.. Please refer to the individual notebooks for more details on the specific datasets used.

FYI, all the notebooks in this repository can be run by just pulling the repository and running on Google Colab, Kaggle, or a good GPU computer.

Feel free to customize the above format to fit your specific needs. Don't forget to add the necessary information and details for each of the notebooks in your repository. Good luck with your project!

## Models Used
- Simple RNN
- LSTM
- Bi-directional LSTM
- Encoder-Decoder

### Notebooks
- [Engish-to-German](https://github.com/ni9/Machine-Translation/blob/English-German/English-German/machine_translation_english_to_german.ipynb)
- [German-to-English](https://github.com/ni9/Machine-Translation/blob/English-Italian/English_To_German_To_English/German-English/machine_translation_german_to_english.ipynb)
- [English-to-Italian](https://github.com/ni9/Machine-Translation/blob/English-Italian/English_To_Italian/English_to_Italian.ipynb)
- [Japanese-to-English](https://github.com/ni9/Machine-Translation/blob/ashish/Japanese_To_English/Jpn_to_Eng.ipynb)
- [French-to-English](https://github.com/ni9/Machine-Translation/blob/fr_to_en/machine_translation.ipynb)
- [English-to-French](https://github.com/ni9/Machine-Translation/blob/develop/English_To_French_To_English/machine_translation.ipynb)


## Usage
Each notebook contains detailed instructions for running the respective model. Please refer to the README file included in each model directory for additional information.

##Result
**Model**                            | **Accuracy** | **Precision/BLEU Score** | **Recall/ROUGE Score** | **F1 Score**
------------------------------------|--------------|-------------------------|------------------------|--------------
RNN/GRU                              |   0.84       |         0.53             |        0.54            |   0.54
RNN/GRU with Embedding               |   0.94       |         0.62             |        0.62            |   0.62
LSTM with Embedding                  |   0.94       |         0.61             |        0.61            |   0.61
Bidirectional LSTM                   |   0.97       |         0.68             |        0.68            |   0.68
RNN/GRU based Encoder-Decoder        |   0.69       |         0.36             |        0.39            |   0.37



## Conclusion 
In summary, our project achieved successful multilingual translation. We created models that accurately translate text between different languages, such as German, Italian, French, and Japanese. The results showed promising performance in terms of accuracy and loss metrics. This demonstrates the potential of machine translation in breaking down language barriers and facilitating global communication. Although there is room for further improvement, our project highlights the effectiveness of modern deep-learning techniques in enabling multilingual understanding and connectivity.


### Team
- Nimesh Arora (Team Lead)
- Dikshant Sagar
- Shreyas Teli
- Ashish Gurung
- Safal Rijal
