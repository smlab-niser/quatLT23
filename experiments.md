# How much time forward and backward propagation take separately

Following experiment was carried out on an NVIDA RTX A6000 GPU using mnist dataset on lennet-300-100 model. The following data is averaged over 1000 epochs. Here is the data for different batch sizes:

<!-- | batchsize  | models     | Forward  | Backward |
| ---------- | ---------- | -------- | -------- |
| bs = 2**13 | Real       | 0.447ms  | 1.334ms  |
|            | Quaternion | 17.513ms | 3.484ms  | -->

<table style="text-align:center">
  <thead>
    <tr>
      <th>batchsize</th>
      <th>models</th>
      <th>forward </th>
      <th>backward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>8192</b></td>
      <td><b>Real</b></td>
      <td>3.128ms</td>
      <td>1.625ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>13.213ms</td>
      <td>3.845ms</td>
    </tr>
    <tr>
      <td rowspan="2"><b>4096</b></td>
      <td><b>Real</b></td>
      <td>3.149ms</td>
      <td>2.665ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>16.693ms</td>
      <td>7.080ms</td>
    </tr>
    <tr>
      <td rowspan="2"><b>2048</b></td>
      <td><b>Real</b></td>
      <td>1.876ms</td>
      <td>6.007ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>24.539ms</td>
      <td>13.378ms</td>
    </tr>
    <tr>
      <td rowspan="2"><b>1024</b></td>
      <td><b>Real</b></td>
      <td>2.773ms</td>
      <td>8.011ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>37.976ms</td>
      <td>24.690ms</td>
    </tr>
    <tr>
      <td rowspan="2"><b>512</b></td>
      <td><b>Real</b></td>
      <td>5.839ms</td>
      <td>17.781ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>44.843ms</td>
      <td>37.602ms</td>
    </tr>
    <tr>
      <td rowspan="2"><b>256</b></td>
      <td><b>Real</b></td>
      <td>10.550ms</td>
      <td>26.472ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>62.014ms</td>
      <td>66.589ms</td>
    </tr>
    <tr>
      <td rowspan="2"><b>128</b></td>
      <td><b>Real</b></td>
      <td>21.356ms</td>
      <td>51.359ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>106.219ms</td>
      <td>130.599ms</td>
    </tr>
    <tr>
      <td rowspan="2"><b>64</b></td>
      <td><b>Real</b></td>
      <td>40.802ms</td>
      <td>99.218ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>219.722ms</td>
      <td>285.788ms</td>
    </tr>
    <tr>
      <td rowspan="2"><b>32</b></td>
      <td><b>Real</b></td>
      <td>89.829ms</td>
      <td>236.377ms</td>
    </tr>
    <tr>
      <td><b>Quart</b></td>
      <td>447.518ms</td>
      <td>605.504ms</td>
    </tr>    
  </tbody>
</table>