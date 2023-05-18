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
    <td>0.70</td>
    <td>0.37</td>
    <td>0.45</td>
    <td>0.41</td>
  </tr>
  <tr>
    <td>Simple RNN/GRU with Embedding</td>
    <td><b>0.81</b></td>
    <td>0.56</td>
    <td>0.67</td>
    <td>0.61</td>
  </tr>
  <tr>
    <td>LSTM with Embedding</td>
    <td><b>0.81</b></td>
    <td><b>0.58</b></td>
    <td><b>0.69</b></td>
    <td><b>0.63</b></td>
  </tr>
  <tr>
    <td>Bidirectional-LSTM</td>
    <td><b>0.81</b></td>
    <td>0.56</td>
    <td>0.66</td>
    <td>0.60</td>
  </tr>
  <tr>
    <td>RNN/GRU based Encoder-Decoder</td>
    <td>0.70</td>
    <td>0.35</td>
    <td>0.41</td>
    <td>0.38</td>
  </tr>
</tbody>
</table>
