# How much time forward and backward propagation take separately

Following experiment was carried out on an NVIDA RTX A6000 GPU using mnist dataset on lennet-300-100 model. The following data is averaged over 1000 epochs.

| A6000 | Real | Quaternion |
| --- | --- | --- |
| Forward  | 0.447ms | 17.513ms |
| Backward | 1.334ms |  3.484ms |
