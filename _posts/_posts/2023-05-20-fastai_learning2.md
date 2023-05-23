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


### Step 3: design or describe the loss function



