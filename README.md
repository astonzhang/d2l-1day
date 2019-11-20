# Dive into Deep Learning in 1 Day

Last updated：|today|

## Information

- Speakers: Rachel Hu, Zhi Zhang
- Date & Time: 11/21/2019, 9AM - 5PM
- Location: LAX10

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

## Prerequisites

You should have some basic knowledge of
[Linear Algebra](http://numpy.d2l.ai/chapter_preliminaries/scalar-tensor.html),
[Calculus](http://numpy.d2l.ai/chapter_preliminaries/calculus.html),
[Probability](http://numpy.d2l.ai/chapter_preliminaries/probability.html), and
[Python](https://learnpythonthehardway.org/) (here's
[another book](https://www.diveinto.org/python3/table-of-contents.html) to learn
Python). Moreover, you should have some experience with
[Jupyter](https://jupyter.org/) notebooks, or with
[SageMaker](http://aws.amazon.com/sagemaker) notebooks. To run things
on (multiple) GPUs you need access to a GPU server, such as the
[P2](https://aws.amazon.com/ec2/instance-types/p2/),
[G3](https://aws.amazon.com/ec2/instance-types/g3/), or
[P3](https://aws.amazon.com/ec2/instance-types/p3/)
instances.


## Syllabus

- This course relies heavily on the
  [Dive into Deep Learning](http://numpy.d2l.ai) book. There's a lot more
  detail in the book (notebooks, examples, math, applications).
- The crash course will get you started. For more information also see [other
  courses and tutorials](http://courses.d2l.ai) based on the book.
- All notebooks below are availabe at [d2l-ai/1day-notebooks](https://github.com/d2l-ai/1day-notebooks), which contains instructions how to setup the running environments.


| Time | Topics |
| --- | --- |
| 9:00---9:20 | Setup clinic for laptops |
| 9:20---10:40 | [Part 1: Deep learning basic](#part-1-deep-learning-basic) |
| 10:40---11:00 | Coffee break |
| 11:00---12:30 | [Part 2: Convolutional neural networks](#part-2-convolutional-neural-networks) |
| 12:30---1:30 | Lunch break |
| 13:30---15:00 | [Part 3: Best practices](#part-3-performance) |
| 15:00---15:30 | Coffee break |
| 15:30---17:00 | [Part 4: Recurrent neural networks](#part-4-recurrent-neural-networks) |


### Part 1: Deep Learning Basic

**Slides**: [[keynote]](slides/DL_basics.key), [[pdf]](slides/DL_basics.pdf)

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

### Part 2: Convolutional Neural Networks

**Slides**: [[keynote]](slides/CNN.key), [[pdf]](slides/CNN.pdf)

**Notebooks**:

1. GPUs                                          [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb#/)
1. Convolutions                                  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/2-conv-layer.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/2-conv-layer.ipynb#/)
1. Pooling                                       [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/3-pooling.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/3-pooling.ipynb#/)
1. Convolutional Neural Networks (LeNet)         [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/4-lenet.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/4-lenet.ipynb#/)
1. Deep Convolutional Neural Networks (AlexNet)  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/5-alexnet.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/5-alexnet.ipynb#/)
1. Networks Using Blocks (VGG)                   [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/6-vgg.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/6-vgg.ipynb#/)
1. Inception Networks (GoogLeNet)                [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/7-googlenet.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/7-googlenet.ipynb#/)
1. Residual Networks (ResNet)                    [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/8-resnet.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/8-resnet.ipynb#/)

### Part 3: Best Practices

**Slides**: [[keynote]](slides/Performance.key), [[pdf]](slides/Performance.pdf)

**Notebooks**:

1. A Hybrid of Imperative and Symbolic Programming    [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/1-hybridize.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/1-hybridize.ipynb#/)
1. Multi-GPU Computation Implementation from Scratch  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/2-multiple-gpus.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/2-multiple-gpus.ipynb#/)
1. Concise Implementation of Multi-GPU Computation    [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/3-multiple-gpus-gluon.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/3-multiple-gpus-gluon.ipynb#/)
1. Fine Tuning                                        [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/4-fine-tuning.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/4-fine-tuning.ipynb#/)


### Part 4: Recurrent Neural Networks

**Slides**: [[keynote]](slides/RNN.key), [[pdf]](slides/RNN.pdf)

**Notebooks**:

1. Text Preprocessing                                        [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/1-text-preprocessing.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/1-text-preprocessing.ipynb#/)
1. Implementation of Recurrent Neural Networks from Scratch  [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/2-rnn-scratch.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/2-rnn-scratch.ipynb#/)
1. Concise Implementation of Recurrent Neural Networks       [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/3-rnn-gluon.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/3-rnn-gluon.ipynb#/)
1. Gated Recurrent Units (GRU)                               [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/4-gru.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/4-gru.ipynb#/)
1. Long Short Term Memory (LSTM)                             [[ipynb]](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/5-lstm.ipynb)  [[slides]](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/5-lstm.ipynb#/)


# Build Mini Amazon-Go in 1 Day Hackathon

[Amazon Go](https://www.amazon.com/b?ie=UTF8&node=16008589011) is a new kind of store with no checkout required. With the "Just Walk Out Shopping" experience, simply use the Amazon Go app to enter the store, take any of the products, and go! No lines, no checkout. A little later, you will receive a receipt on the app with a charge to your Amazon account. This checkout-free shopping experience is made possible by the same types of technologies used in self-driving cars: computer vision, sensor fusion, and deep learning. The technology automatically detects when products are taken from or returned to the shelves and keeps track of them in a virtual cart. When the customers done shopping, they can just leave the store without checking out. At the moment they walk out of the door, the technology will capture the photos, detect the products, automatically make the transactions, and send the cutomers the receipts.

Now, do you want to be involved in deep learning and contribute to the bleeding edge of technology of computer vision? In this Hackathon, we will empower you to team up and build a "Mini Amazon Go" from scratch. Complimentary food and drink are served (just don't dine and dash).

## Hackathon Information

- Instructor: Zhi Zhang, Rachel Hu
- Date: 11/22/2019, 1 - 5PM
- Location: LAX10

## Hackathon Prerequisites

A laptop(with modern browser) is sufficient to access all resources in order to take part in the hackathon!


## Contents

Overview of the hackathon can be viewed prior to this event: [[keynote]](slides/MiniAmazonGo.key), [[pdf]](slides/MiniAmazonGo.pdf)

## Awards

Participants will be granted with a [phone tool icon](https://phonetool.amazon.com/awards/96139/award_icons/108886). The winners will win a full set of "mini Amazon Go" hardwares, so you can play with it any time.
