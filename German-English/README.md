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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-fymr">Model</th>
    <th class="tg-fymr">Accuracy</th>
    <th class="tg-fymr">Precision/BLEU Score</th>
    <th class="tg-fymr">Recall/ROUGE Score</th>
    <th class="tg-fymr">F1-Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Simple RNN/GRU</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">0.7041</span></td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">0.372056</span></td>
    <td class="tg-0pky">0.447902</td>
    <td class="tg-0pky">0.406180</td>
  </tr>
  <tr>
    <td class="tg-0pky">Simple RNN/GRU with Embedding</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">0.8061</span><br></td>
    <td class="tg-0pky">0.558837</td>
    <td class="tg-0pky">0.668223</td>
    <td class="tg-0pky">0.608150</td>
  </tr>
  <tr>
    <td class="tg-0pky">LSTM with Embedding</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">0.8054</span></td>
    <td class="tg-fymr">0.577381</td>
    <td class="tg-fymr">0.689327</td>
    <td class="tg-fymr">0.627894</td>
  </tr>
  <tr>
    <td class="tg-0pky">Bidirectional-LSTM</td>
    <td class="tg-fymr"><span style="font-weight:bold;font-style:normal">0.8140</span></td>
    <td class="tg-0pky">0.555476</td>
    <td class="tg-0pky">0.663686</td>
    <td class="tg-0pky">0.604267</td>
  </tr>
  <tr>
    <td class="tg-0pky">RNN/GRU based Encoder-Decoder</td>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">0.6966</span></td>
    <td class="tg-0pky">0.346580</td>
    <td class="tg-0pky">0.409116</td>
    <td class="tg-0pky">0.375073</td>
  </tr>
</tbody>
</table>
