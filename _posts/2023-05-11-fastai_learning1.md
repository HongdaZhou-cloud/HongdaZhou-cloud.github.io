# Fastai Learning Blog (1)


## Neural network and deep learning
### Neural network:

Neural networks, specifically, artificial neural networks (ANNs)—mimic the human brain through a set of algorithms. At a basic level, a neural network is comprised of four main components: inputs, weights, a bias or threshold, and an output. Similar to linear regression, the algebraic formula would look something like this:

<img width="390" alt="2023-05-23_113145" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/6c045528-f714-470e-9e1b-b52cc846b6cc">

Here, "x" presents inputs; "w" is the weight for the respective inputs and "bias" is the bias or threshold.

If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network. Neural networks tend to have multiple “hidden” layers as part of deep learning algorithms, and each hidden layer has its own activation function.

### Deep learning:

The “deep” in deep learning is referring to the depth of layers in a neural network. A neural network that consists of more than three layers—which would be inclusive of the inputs and the output—can be considered a deep learning algorithm. This is generally represented using the following diagram:

<img width="374" alt="2023-05-23_112156" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/bf2b688c-e4d9-45f3-ae43-a8b98aa8aed8">


## Fastai introduction

fastai is a deep learning library based on Python language and PyTorch library. It provides components that can not only quickly provide results in standard deep learning domains but also allow it to be mixed to build new approaches. The resouce in detail can be found on the fastai website in https://docs.fast.ai/.


## First lesson: is it a bird?

The basic steps of this lesson are:

1. Use DuckDuckGo to search for images of "bird photos" and "forest photos"
1. Fine-tune a pretrained neural network to recognise these two groups
1. Try running this model on a picture of a bird and see if it works.

### Step 1: Search for images

The bird and forest photoes are required to be searched. Firstly, we start by getting URLs from a search, and then save each group of photos to a different folder.


### Step 2: Train our model

<img width="826" alt="2023-05-23_111540" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/2a60718d-977a-4c15-9061-e5af0538f1ef">


### Step 3: Use our model 

<img width="534" alt="2023-05-23_111136" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/d8a9cfac-e08e-4c1b-b853-07e397f6be0a">
