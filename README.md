# Dive into Deep Learning in 1 Day

Last updated：|today|

## Information

- Speaker： [Alex Smola](https://alex.smola.org/)
- Date： [TBA]
- Location：[TBA]

## Overview

Did you ever want to find out about deep learning but didn't have time
to spend months? New to machine learning? Do you want to build image
classifiers, NLP apps, train on many GPUs or even on many machines?
If you're an engineer or data scientist, this course is for you.
This is about the equivalent of a Coursera course, all packed into one
day. The course consists of four segments of 90 minutes each.

1. Deep Learning Basics
1. Convolutional Neural Networks for computer vision
1. Best practices (GPUs, Parallelization, Fine Tuning, Transfer Learning)
1. Recurrent Neural Networks for natural language (RNN, LSTM, GRU)



## Syllabus

- This course relies heavily on the
  [Dive into Deep Learning](http://numpy.d2l.ai) book. There's a lot more
  detail in the book (notebooks, examples, math, applications).
- The crash course will get you started. For more information also see [other
  courses and tutorials](http://courses.d2l.ai) based on the book.
- All notebooks below are availabe at [d2l-ai/1day-notebooks](https://github.com/d2l-ai/1day-notebooks), which contains instructions how to setup the running environments.


| Time | Topics |
| --- | --- |
| 8:00---9:00 | Setup clinic for laptops |
| 9:00---10:30 | [Part 1: Deep learning basic](#id5) |
| 10:30---11:00 | Coffee break |
| 11:00---12:30 | [Part 2: Convolutional neural networks](#id6) |
| 12:30---2:00 | Lunch break |
| 2:00---3:30 | [Part 3: Performance](#id7) |
| 3:30---4:00 | Coffee break |
| 4:00---5:30 | [Part 4: Recurrent neural networks](#id8) |


### Part 1: Deep Learning Basic

**Slides**: [[keynote]](slides/Part-1.key), [[pdf]](slides/Part-1.pdf)

**Notebooks**:

1. Data Manipulation with Ndarray  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/1-ndarray.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/1-ndarray.ipynb#/)
1. Automatic Differentiation  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/2-autograd.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/2-autograd.ipynb#/)
1. Linear Regression Implementation from Scratch  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/3-linear-regression-scratch.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/3-linear-regression-scratch.ipynb#/)
1. Concise Implementation of Linear Regression  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/4-linear-regression-gluon.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/4-linear-regression-gluon.ipynb#/)
1. Image Classification Data (Fashion-MNIST)  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/5-fashion-mnist.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/5-fashion-mnist.ipynb#/)
1. Implementation of Softmax Regression from Scratch  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/6-softmax-regression-scratch.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/6-softmax-regression-scratch.ipynb#/)
1. Concise Implementation of Softmax Regression  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/7-softmax-regression-gluon.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/7-softmax-regression-gluon.ipynb#/)
1. Implementation of Multilayer Perceptron from Scratch  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/8-mlp-scratch.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/8-mlp-scratch.ipynb#/)
1. Concise Implementation of Multilayer Perceptron  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/9-mlp-gluon.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/9-mlp-gluon.ipynb#/)

### Part 2: Convolutional neural networks

**Slides**: [[keynote]](slides/Part-2.key), [[pdf]](slides/Part-2.pdf)

**Notebooks**:

1. GPUs                                          [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb#/)
1. Convolutions                                  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/2-conv-layer.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/2-conv-layer.ipynb#/)
1. Pooling                                       [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/3-pooling.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/3-pooling.ipynb#/)
1. Convolutional Neural Networks (LeNet)         [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/4-lenet.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/4-lenet.ipynb#/)
1. Deep Convolutional Neural Networks (AlexNet)  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/5-alexnet.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/5-alexnet.ipynb#/)
1. Networks Using Blocks (VGG)                   [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/6-vgg.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/6-vgg.ipynb#/)
1. Inception Networks (GoogLeNet)                [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/7-googlenet.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/7-googlenet.ipynb#/)
1. Residual Networks (ResNet)                    [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/8-resnet.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/8-resnet.ipynb#/)

### Part 3: Performance

**Slides**: [[keynote]](slides/Part-3.key), [[pdf]](slides/Part-3.pdf)

**Notebooks**:

1. A Hybrid of Imperative and Symbolic Programming    [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/1-hybridize.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/1-hybridize.ipynb#/)
1. Multi-GPU Computation Implementation from Scratch  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/2-multiple-gpus.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/2-multiple-gpus.ipynb#/)
1. Concise Implementation of Multi-GPU Computation    [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/3-multiple-gpus-gluon.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/3-multiple-gpus-gluon.ipynb#/)
1. Fine Tuning                                        [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/4-fine-tuning.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/4-fine-tuning.ipynb#/)


### Part 4: Recurrent neural networks

**Slides**: [[keynote]](slides/Part-4.key), [[pdf]](slides/Part-4.pdf)

**Notebooks**:

1. Text Preprocessing                                        [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/1-text-preprocessing.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/1-text-preprocessing.ipynb#/)
1. Implementation of Recurrent Neural Networks from Scratch  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/2-rnn-scratch.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/2-rnn-scratch.ipynb#/)
1. Concise Implementation of Recurrent Neural Networks       [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/3-rnn-gluon.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/3-rnn-gluon.ipynb#/)
1. Gated Recurrent Units (GRU)                               [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/4-gru.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/4-gru.ipynb#/)
1. Long Short Term Memory (LSTM)                             [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/5-lstm.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/5-lstm.ipynb#/)

