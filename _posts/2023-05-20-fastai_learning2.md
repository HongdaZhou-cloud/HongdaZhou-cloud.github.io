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

A loss function, also known as an objective function or a cost function, is a mathematical function used to measure the error or discrepancy between the predicted output of a machine learning model and the true or target output. The purpose of a loss function is to quantify how well the model is performing during training and to guide the model's optimization process. 

An ideal loss function should be differentiable, continuous, and have well-behaved gradients. This ensures that optimization algorithms can efficiently find the parameters that minimize the loss. It's also important to choose an appropriate loss function that aligns with the problem at hand and the desired behavior of the model. The selection of a loss function can have a significant impact on the model's final performance.

As discussed in Fastai Learning Blog (1), the "loss" in the image below just refers to the loss function.

<img width="826" alt="2023-05-23_111540" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/2a60718d-977a-4c15-9061-e5af0538f1ef">

Here, we use the inputs and weights to get the final results, and then calculate the loss function to go back and keeping adjust the weights. The model training is not finished until we get very small loss.

The choice of a loss function depends on the specific task at hand. Different machine learning tasks, such as classification, regression, or generative modeling, require different types of loss functions.

In this project, as we did not select a loss or optimizer function before, fastai already tried to choose the best selection for us. We can check the loss function by calling loss_func directly.

```python
learn.loss_func
```
It can be known that the loss function applied here is "FlattenedLoss of CrossEntropyLoss()".

"FlattenedLoss of CrossEntropyLoss()" operates on flattened inputs and then utilizes the CrossEntropyLoss function for calculating the loss.

By using the FlattenedLoss wrapper, we can ensure that the CrossEntropyLoss function is applied correctly to flattened input data. This allows us to use the models that expect flattened input tensors while still benefiting from the functionality and computation of the CrossEntropyLoss function.
