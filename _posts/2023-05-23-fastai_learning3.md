# Fastai Learning Blog (3)


## Build my own model to identify ten animals

This blog is the based on the results gotten from "Fastai Learning Blog (2)". Two methods: Confusion matrices and t-SNE, are introduced here in detail to analyse the data.

### Step 4: Analyse the results

#### Confusion matrices:
A confusion matrix is a tabular representation that summarizes the performance of a classification model by comparing the predicted labels with the true labels. It shows the counts of true positives, true negatives, false positives, and false negatives. Each row in the matrix corresponds to the actual class labels, while each column represents the predicted class labels.

This is a typical confusion matrix:

<img width="380" alt="2023-05-23_212831" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/1dd2bd38-4438-4aa1-90d3-c067fa30d847">

The column labels “Actually Positive” and “Actually Negative” represent the ground-truth labels in data set, and the row labels “Predicted Positive” and “Predicted Negative” refer to the model’s predictions, i.e. what model thinks the label is.

What's more, the values in the confusion matrix have the following meanings:

|Value| Meaning |
|-|-|
|True Positive (TP)| The number of samples that are correctly predicted as positive. |
|True Negative (TN)| The number of samples that are correctly predicted as negative. |
|False Positive (FP)| The number of samples that are incorrectly predicted as positive. |
|False Negative (FN)| The number of samples that are incorrectly predicted as negative. | 

The confusion matrix provides valuable information about the performance of a classification model. It allows us to assess the accuracy, precision, recall, and other evaluation metrics derived from these counts.

Specially, for the code in fastai:

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```
We can get the result:

<img width="287" alt="2023-05-23_215512" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/feeb7c0e-e809-4e63-9905-264898d45e96">

#### t-SNE:
t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that is commonly used for visualizing high-dimensional data in a lower-dimensional space, typically 2D or 3D. It aims to preserve the local structure of the data points while revealing global patterns and relationships.

The t-SNE algorithm works by constructing a probability distribution over pairs of high-dimensional data points, then constructing a similar probability distribution over pairs of low-dimensional points. It minimizes the divergence between these two distributions using gradient descent, adjusting the positions of the low-dimensional points iteratively.

After obtaining the t-SNE embedding, we can visualize it using plotting libraries to explore the structure and relationships of data in a lower-dimensional space.

Specially, for the code in fastai:

```python
# get the input data
data, _ = learn.get_preds(dl=dls.valid)

# low-dimensional embedding
tsne = TSNE(n_components=2)
embedding = tsne.fit_transform(data)

# get the target labels
targets = [str(dls.vocab[x[1]]) for x in dls.valid_ds]
target_labels = LabelEncoder().fit_transform(targets)

# plot the result
plt.scatter(embedding[:, 0], embedding[:, 1], c=target_labels)
plt.title('t-SNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
```
We can get the result:








