# Fastai Learning Blog (3)


## Build my own model to identify ten animals

This blog is the based on the results gotten from "Fastai Learning Blog (2)". Two methods: Confusion matrices and t-SNE are introduced here in detail to analyse the data.

### Step 4: Analyse the results

#### Confusion matrices:
A confusion matrix is a tabular representation that summarizes the performance of a classification model by comparing the predicted labels with the true labels. It shows the counts of true positives, true negatives, false positives, and false negatives. Each row in the matrix corresponds to the actual class labels, while each column represents the predicted class labels.

This is a typical confusion matrix:

<img width="255" alt="2023-05-23_212831" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/1dd2bd38-4438-4aa1-90d3-c067fa30d847">

The column labels “Actually Positive” and “Actually Negative” refer to the ground-truth labels in your data set, i.e. whether a handwritten digit is truly a 1 or a 0, whether a patient was truly diagnosed with a disease (1) or not (0), whether a chest x-ray actually shows pneumonia (1) or not (0), etc.

The row labels “Predicted Positive” and “Predicted Negative” refer to your model’s predictions, i.e. what your model thinks the label is.

True Positive (TP): The number of samples that are correctly predicted as positive.
True Negative (TN): The number of samples that are correctly predicted as negative.
False Positive (FP): The number of samples that are incorrectly predicted as positive (also known as a Type I error).
False Negative (FN): The number of samples that are incorrectly predicted as negative (also known as a Type II error).

The confusion matrix provides valuable information about the performance of a classification model. It allows you to assess the accuracy, precision, recall, and other evaluation metrics derived from these counts. It is particularly useful when dealing with imbalanced datasets, where the distribution of classes is uneven.

#### t-SNE:
t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that is commonly used for visualizing high-dimensional data in a lower-dimensional space, typically 2D or 3D. It aims to preserve the local structure of the data points while revealing global patterns and relationships.

The t-SNE algorithm works by constructing a probability distribution over pairs of high-dimensional data points, then constructing a similar probability distribution over pairs of low-dimensional points. It minimizes the divergence between these two distributions using gradient descent, adjusting the positions of the low-dimensional points iteratively.

After obtaining the t-SNE embedding, we can visualize it using plotting libraries to explore the structure and relationships of data in a lower-dimensional space.










