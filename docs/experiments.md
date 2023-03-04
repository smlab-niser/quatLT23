# Various Experiments Performed

## How much time forward and backward propagation take separately

Following experiment was carried out on an NVIDA RTX A6000 GPU using mnist dataset on lennet-300-100 model. The following data is averaged over 1000 epochs. Here is the data for different batch sizes:

<!-- | batchsize  | models     | Forward | Backward |
| ---------- | ---------- | -------- | ------- |
| bs = 2**13 | Real       | 0.447ms  | 1.334ms |
|            | Quaternion | 17.513ms | 3.484ms | -->

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
            <td><b>Quat</b></td>
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
            <td><b>Quat</b></td>
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
            <td><b>Quat</b></td>
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
            <td><b>Quat</b></td>
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
            <td><b>Quat</b></td>
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
            <td><b>Quat</b></td>
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
            <td><b>Quat</b></td>
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
            <td><b>Quat</b></td>
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
            <td><b>Quat</b></td>
            <td>447.518ms</td>
            <td>605.504ms</td>
        </tr>
    </tbody>
</table>

## Determination and analysis of quaternion model's slow pace

* Analyzed the QuaternionTensor codes and found out that these [lines](https://github.com/smlab-niser/QuatLT23/blob/main/htorch/quaternion.py#L466) are taking a lot of time as it is copying the data from cuda to CPU every time.
* The three steps of forward propagation of a quaternion layer were:

    1. building 4*4 quaternion to real matrix (w)
    2. Finding wx+b (applying linear function)
    3. typecasting the output of step 2 to a quaternion tensor.

    see these [lines](https://github.com/smlab-niser/QuatLT23/blob/main/htorch/layers.py#L263-L270).
  We checked the time taken by these three steps separately. We found that step 1 took around 0.5ms, step 2 took 1ms and step 3 took the rest of the 28.5ms out of the total 30ms taken by forward propagation. The typecasting step is taking around 95% of the time.
* We removed the typecast, and it did not result in any error. Furthermore, it gave us the same accuracy results with a total time consumption of 2.23ms (for the forward propagation of a quaternion layer), very close to that of a real model (1.235ms).
* The final and most useful attempt is given in the following section.
## Before and after changing `q.cpu()` to `q.cuda()`

Changes were made in the returning line of the `QuaternionTensor.__new__()` method in the file [`htorch/quaternion.py`](../htorch/quaternion.py#469). THe change was that instead of removing the typecast, we found the exact line in the class's `__new__()` in which `q.cpu()` was used to send the data to CPU. We replaced it with `q.cuda()` and made it faster. 

- __Model used:__ lenet-300-100
- __Dataset:__ mnist
- __Batch size:__ 8192

| Model | Speed (it/s) |
| ----- | ----------- |
| Real  | 70 - 75 |
| Quat (before) | 20 -25 |
| Quat (after)  | 55 - 60 |

Furthermore, we repeated the batchsize experiment from earlier section with the improved Quaternion models.

## Batchsize experiment with improved quaternion model

- __Model used:__ lenet-300-100
- __Dataset:__ mnist

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
            <td>2.833ms</td>
            <td>1.224ms</td>
            </tr>
<tr>
            <td><b>Quat</b></td>
            <td>1.741ms</td>
            <td>4.409ms</td>
            </tr>
<tr>
            <td rowspan="2"><b>4096</b></td>
            <td><b>Real</b></td>
            <td>2.824ms</td>
            <td>1.982ms</td>
            </tr><tr>
            <td><b>Quat</b></td>
            <td>2.479ms</td>
            <td>4.425ms</td>
            </tr><tr>
            <td rowspan="2"><b>2048</b></td>
            <td><b>Real</b></td>
            <td>1.712ms</td>
            <td>5.231ms</td>
            </tr><tr>
            <td><b>Quat</b></td>
            <td>4.904ms</td>
            <td>8.499ms</td>
            </tr><tr>
            <td rowspan="2"><b>1024</b></td>
            <td><b>Real</b></td>
            <td>2.488ms</td>
            <td>6.930ms</td>
            </tr><tr>
            <td><b>Quat</b></td>
            <td>9.589ms</td>
            <td>17.100ms</td>
            </tr><tr>
            <td rowspan="2"><b>512</b></td>
            <td><b>Real</b></td>
            <td>4.959ms</td>
            <td>13.531ms</td>
            </tr><tr>
            <td><b>Quat</b></td>
            <td>19.205ms</td>
            <td>33.230ms</td>
            </tr><tr>
            <td rowspan="2"><b>256</b></td>
            <td><b>Real</b></td>
            <td>10.350ms</td>
            <td>26.507ms</td>
            </tr><tr>
            <td><b>Quat</b></td>
            <td>38.441ms</td>
            <td>65.430ms</td>
            </tr><tr>
            <td rowspan="2"><b>128</b></td>
            <td><b>Real</b></td>
            <td>20.993ms</td>
            <td>51.605ms</td>
            </tr><tr>
            <td><b>Quat</b></td>
            <td>75.729ms</td>
            <td>129.030ms</td>
            </tr><tr>
            <td rowspan="2"><b>64</b></td>
            <td><b>Real</b></td>
            <td>40.321ms</td>
            <td>99.586ms</td>
            </tr><tr>
            <td><b>Quat</b></td>
            <td>148.626ms</td>
            <td>253.968ms</td>
            </tr><tr>
            <td rowspan="2"><b>32</b></td>
            <td><b>Real</b></td>
            <td>80.087ms</td>
            <td>200.908ms</td>
            </tr><tr>
            <td><b>Quat</b></td>
            <td>297.166ms</td>
            <td>510.735ms</td>
            </tr>
    </tbody>
</table>