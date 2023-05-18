# Machine Translation

## German To English Translation [ <a href="https://github.com/dikshantsagar">Dikshant Sagar</a> ]

### Dataset Files

- `small_vocab_de` : German Text File containing one sentence in each line.
- `small_vocab_en` : English Text File containing corresponding parallel sentence in each line.


### Models Implemented

- `Simple RNN/GRU Based Neural Network`
- ` Simple RNN/GRU Based Neural Network with Embedding Layer`
- ` LSTM Based Neural Network with Embedding Layer`
- ` Bidirectional LSTM Neural Network`
- ` RNN/GRU Encoder-Decoder Based Neural Network`


### Metrics Evaluated

- `Neural Network's Cross-Entropy Loss`
- `Accuracy`
- `Precision/BLEU Score`
- `Recall/ROUGE Score`
- `F1-Score`


### Results

<table class="tg">
<thead>
  <tr>
    <th><b>Model</b></th>
    <th><b>Accuracy</b></th>
    <th><b>Precision/BLEU Score</b></th>
    <th><b>Recall/ROUGE Score</b></th>
    <th><b>F1-Score</b></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Simple RNN/GRU</td>
    <td>0.7041</td>
    <td>0.372056</td>
    <td>0.447902</td>
    <td>0.406180</td>
  </tr>
  <tr>
    <td>Simple RNN/GRU with Embedding</td>
    <td>0.8061</td>
    <td>0.558837</td>
    <td>0.668223</td>
    <td>0.608150</td>
  </tr>
  <tr>
    <td>LSTM with Embedding</td>
    <td>0.8054</td>
    <td><b>0.577381</b></td>
    <td><b>0.689327</b></td>
    <td><b>0.627894</b></td>
  </tr>
  <tr>
    <td>Bidirectional-LSTM</td>
    <td><b>0.8140</b></td>
    <td>0.555476</td>
    <td>0.663686</td>
    <td>0.604267</td>
  </tr>
  <tr>
    <td>RNN/GRU based Encoder-Decoder</td>
    <td>0.6966</td>
    <td>0.346580</td>
    <td>0.409116</td>
    <td>0.375073</td>
  </tr>
</tbody>
</table>
