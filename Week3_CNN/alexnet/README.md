PASCAL VOC AlexNet tutorial for Places10 custum dataset.

Implemented original AlexNet which has,
- local response normalizer
- overlapping pooling
- dropout
- Gaussian Distribution initializer for kernel filters
- zeros/ones initializer for biases
- cross entropy loss and SGD optimizer
- random flip preprocessing

Modified and Added some features
- TFRecords input pipelines
- random crop (originally boosts data by cropping 10 times)
- slightly smaller batch size 128 --> 100
- 1000 --> 10 classes
- FC6, FC7 layers depth: 4096, 4096 --> 512, 128
- no 'weight decay'
- learning rate decay: step function --> exponential decay

- performance of initial release version Places10_AlexNet:

  165641ms, Epoch: 15 Validation accuracy = 85.0%

  Epoch: 30 Validation accuracy = 86.9%

  Epoch: 50 Validation accuracy = 87.4%

  Epoch: 75 Validation accuracy = 87.7%

- with Batch Normalization,

  169425ms, Epoch: 48 Validation accuracy = 87.7%

- with BN, and weight decay

  632974ms, Epoch: 40 Validation accuracy = 86.1%

- with Conv Factorization + 3 additional 3x3 conv layers, and 3 Resnets,

  391296ms, Epoch: 10 Validation accuracy = 88.3%

  Epoch: 14 Validation accuracy = 88.6%

  Epoch: 25 Validation accuracy = 89.1%

- with Conv Factorization + 3 additional 3x3 conv layers, and 1 Resnets, 2 Inception-Resnets,

  358927ms, Epoch: 19 Validation accuracy = 88.6%

  Epoch: 35 Validation accuracy = 89.1%

  Epoch: 36 Validation accuracy = 89.2%

