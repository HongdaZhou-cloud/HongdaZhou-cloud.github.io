# Fastai Learning Blog (1)


## Neural network and deep learning
### Neural network:

Neural networks, specifically, artificial neural networks (ANNs)—mimic the human brain through a set of algorithms. At a basic level, a neural network is comprised of four main components: inputs, weights, a bias or threshold, and an output. Similar to linear regression, the algebraic formula would look something like this:

<img width="390" alt="2023-05-23_113145" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/6c045528-f714-470e-9e1b-b52cc846b6cc">

Here, "x" presents inputs; "w" is the weight for the respective inputs and "bias" is the bias or threshold.

If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network. Neural networks tend to have multiple “hidden” layers as part of deep learning algorithms, and each hidden layer has its own activation function.

### Related to deep learning:

The “deep” in deep learning is referring to the depth of layers in a neural network. A neural network that consists of more than three layers—which would be inclusive of the inputs and the output—can be considered a deep learning algorithm. This is generally represented using the following diagram:

<img width="374" alt="2023-05-23_112156" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/bf2b688c-e4d9-45f3-ae43-a8b98aa8aed8">


## Fastai introduction

fastai is a deep learning library based on Python language and PyTorch library. It provides components that can not only quickly provide results in standard deep learning domains but also allow it to be mixed to build new approaches. The resouce in detail can be found on the fastai website in https://docs.fast.ai/.


## First fastai lesson: is it a bird?

The basic steps of this lesson are:

1. Use DuckDuckGo to search for images of "bird photos" and "forest photos"
1. Fine-tune a pretrained neural network to recognise these two groups
1. Try running this model on a picture of a bird and see if it works.

### Step 1: Search for images

The bird and forest photoes are required to be searched. Firstly, we start by getting URLs from a search, and then save each group of photos to a different folder.

```python
from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```
<img width="191" alt="2023-05-23_165010" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/7cb9c0e3-aa52-450b-8511-642585449b6b">

```python
download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)
```

<img width="189" alt="2023-05-23_165022" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/e2c17980-ca83-474f-bb32-46b8928bb1ec">

### Step 2: Train our model

#### Build data set:
To train a model, we'll need `DataLoaders`, which is an object that contains a *training set* (the images used to create a model) and a *validation set* (the images used to check the accuracy of a model -- not used during training). In `fastai` we can create that easily using a `DataBlock`, and view sample images from it:

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=20)
```
Here is the meaning of each DataBlock parameter (see https://docs.fast.ai/data.load.html in detail):

|Parameter| Explanation |
|-|-|
| blocks=(ImageBlock, CategoryBlock) | The inputs to our model are images, and the outputs are categories (in this case, "bird" or "forest"). |
| get_items=get_image_files | To find all the inputs to our model, run the get_image_files function (which returns a list of all image files in a path). |
| splitter=RandomSplitter(valid_pct=0.2, seed=42) | Split the data into training and validation sets randomly, using 20% of the data for the validation set. |
| get_y=parent_label | The labels (y values) is the name of the parent of each file (i.e. the name of the folder they're in, which will be bird or forest). | 
| item_tfms=[Resize(192, method='squish')] | Before training, resize each image to 192x192 pixels by "squishing" it (as opposed to cropping it). |

#### Train the model:
After that, we can train our model. Here, The computer vision model applied is resnet18. "Fine-tuning" a model means that we're starting with a model someone else has trained using some other dataset, and adjusting the weights a little bit so that the model learns to recognise your particular dataset. 

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(20)
```
It works as below:

<img width="826" alt="2023-05-23_111540" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/2a60718d-977a-4c15-9061-e5af0538f1ef">

Here, we use the inputs and weights to get the final results, and then calculate the loss function to go back and keeping adjust the weights. The model training is not finished until we get very small loss.

### Step 3: Use our model 
Finally, use the model we gotten before to identify if a specific photo is a bird.

```python
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
```
It works as below:

<img width="534" alt="2023-05-23_111136" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/d8a9cfac-e08e-4c1b-b853-07e397f6be0a">

Here, the input is the bird image; model is the trained model we gotten before and the output is the result around whether it is a bird.

Reference:

https://course.fast.ai/Lessons/lesson1.html
