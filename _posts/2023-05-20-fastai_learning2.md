# Fastai Learning Blog (2)


## Build my own model to identify ten animals

After learning the example code in fastai, we can try to build my own model, which basically basic has four steps:

1. Use DuckDuckGo to search for images of ten different kinds of animals;
2. Fine-tune a pretrained neural network to recognise these ten groups;
3. Try to design or describe the loss function;
4. Try to analyse the results with Confusion matrices and t-SNE.

### Step 1: Search for images

Similar as the process in example "is it a bird". Firstly, we try to get URLs from a search, and then save each group of photos to a different folder. Here, "forest" is replaced with other nine kinds of animals. 

```python
searches = 'bird','dog','cat','horse','bear','lion','tiger','fish','panda','kangaroo'
path = Path('bird_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} animals'))
    sleep(10)  # Pause between searches to avoid over-loading server
    resize_images(path/o, max_size=400, dest=path/o)
```
### Step 2: Train our model

#### Build data set:
Also, we'll make a `DataLoaders` that contains a *training set* and a *validation set* with the photoes of ten kinds of animals we searched before.
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
And we get the results of some animals with their labels.

<img width="447" alt="2023-05-23_163718" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/69efa234-fcf2-401e-bed7-3a45fbfd4490">

#### Train the model:
After that, we can train our model.

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(20)
```
We can see the results with the code.

```python
learn.show_results(max_n=16)
```
<img width="365" alt="2023-05-24_001735" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/6786d4e0-4052-4613-b279-84524755e763">

And view the loss images.

```python
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(16)
```
<img width="385" alt="2023-05-24_001712" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/65529cf5-d283-46ed-a56d-cfb9964f578e">

### Step 3: Design or describe the loss function

A loss function (also known as a cost function or objective function) is a mathematical function that quantifies the discrepancy between predicted and true values. It measures the error or loss incurred by a model's predictions and is used to guide the learning or optimization process.

The goal of training a machine learning model is to minimize the loss function, which implies minimizing the discrepancy between predicted and true values. By adjusting the model's parameters or weights, the loss function is minimized, leading to better predictions.

As discussed in Fastai Learning Blog (1), the "loss" in the image below just refers to the loss function.

<img width="826" alt="2023-05-23_111540" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/2a60718d-977a-4c15-9061-e5af0538f1ef">

The choice of a loss function depends on the specific task at hand. Different machine learning tasks, such as classification, regression, or generative modeling, require different types of loss functions.

In this project, as we did not select a loss or optimizer function before, fastai already tried to choose the best selection for us. We can check the loss function by calling loss_func directly.

```python
learn.loss_func
```
It can be known that the loss function applied here is "FlattenedLoss of CrossEntropyLoss()".

"FlattenedLoss of CrossEntropyLoss()" operates on flattened inputs and then utilizes the CrossEntropyLoss function for calculating the loss.

The purpose of the FlattenedLoss is to handle flattened input data, which is often required by models. In this image recognition task, the input data typically has a shape of (batch_size, channels, height, width). However, certain models may expect the input to be flattened into a 2D shape (batch_size, num_features) before applying the loss function.

After flattening the Input, the flattened input and target tensors are passed to the CrossEntropyLoss function, which is a function in PyTorch combining the log_softmax operation with the negative log-likelihood loss. The CrossEntropyLoss computes the cross-entropy loss between the predicted class probabilities and the true class labels. It handles the computation of the loss and incorporates any necessary softmax operations internally. 

By using the FlattenedLoss wrapper, we can ensure that the CrossEntropyLoss function is applied correctly to flattened input data. This allows us to use the models that expect flattened input tensors while still benefiting from the functionality and computation of the CrossEntropyLoss function.
