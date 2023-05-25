# Fastai Learning Blog (4)


## Choose an Optimal Learning Rate

### Learning rate in a neural network

Learning rate in a neural network refers to a hyperparameter that determines the step size at which the model's parameters are updated during the training process. It controls the magnitude of adjustments made to the weights and biases of the network based on the error calculated by the loss function.

Choosing an appropriate learning rate is important for efficient and effective training. It requires finding a balance between fast convergence and stable learning. Setting an optimal learning rate often involves experimentation and fine-tuning.

Four different scenarios will be discussed here:

1. Slow convergence with a smaller learning rate (α1)

2. Oscillating around the minimum with a large learning rate (α2)

3. Oscillating and divergence with a very large learning rate (α3)

4. Properly convergence with the optimal learning rate (α4)

Let’s discuss each scenario in detail.

Scenario 1: Slow convergence with a smaller learning rate

<img width="266" alt="2023-05-25_141517" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/7c9485b2-d1aa-4f61-988a-7c8f635e3515">

This happens when the learning rate is much smaller than the optimal value (α1 << α4). The algorithm takes very small steps to descend the error mountain in order to reach the minimum. So, the convergence happens very slowly and the algorithm takes a lot more time to converge. 

Scenario 2: Oscillating around the minimum with a large learning rate



<img width="418" alt="2023-05-25_141530" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/0ed553f0-ecc1-4ab2-94fe-7940d59a5b63">


This happens when the learning rate is larger than the optimal value (α2 > α4). As the algorithm takes large steps to descend the error mountain, the optimizer overshoots the optimal weight in the first step and it overshoots again in the next step as shown in the figure. The algorithm tries to reach the minimum but it is still far away from the minimum. In the final steps (near the minimum), the optimizer is oscillating around the minimum and never reaches the minimum even if we keep running the algorithm for a long period of time with a high number of epochs. 

Scenario 3: Oscillating and divergence with a very large learning rate




<img width="401" alt="2023-05-25_141542" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/e7aff459-a61c-4cc4-aab8-fc1bf59d04d1">




This happens when the learning rate is much larger than the optimal value (α3 >> α4). Because the learning rate is very large, the algorithm never tries to descend the error mountain. Instead, it overshoots the optimal weight in the first step and continues overshooting in the next steps. After the first few steps, the error starts to increase as the optimizer diverges away from the minimum value. In the final steps, the optimizer is oscillating around a higher error than the started error value. 

Scenario 4: Properly convergence with the optimal learning rate

We should avoid all three cases above, especially the third one. The optimal learning rate occurs somewhere between α1 and α2 (α1 < α4 < α2). With the optimal learning rate, the algorithm reaches the minimum in a short period of time with a considerably fewer number of epochs

In fastai, We can directly use fastai's learn.lr_find() method to find the optimal learning rate.

```python
learn.lr_find()
```

Plotting the loss function against the learning rate yields the following figure:

<img width="440" alt="2023-05-25_134819" src="https://github.com/HongdaZhou-cloud/HongdaZhou-cloud.github.io/assets/132418400/d61e5db4-312b-4cca-9e40-da567a57270a">

It suggests that a learning rate of approximately 0.00010964782268274575 is recommended for training your model.

Once we have identified the optimal learning rate, we can proceed with training our model using that learning rate or adjust it based on our specific requirements.

Reference:
https://towardsdatascience.com/how-to-choose-the-optimal-learning-rate-for-neural-networks-362111c5c783
