# Fastai Learning Blog (4)


## Choose an Optimal Learning Rate

### Learning rate in a neural network

Learning rate in a neural network refers to a hyperparameter that determines the step size at which the model's parameters are updated during the training process. It controls the magnitude of adjustments made to the weights and biases of the network based on the error calculated by the loss function.

One of the challenges of gradient descent is choosing the optimal value for the learning rate, eta (η). The learning rate is perhaps the most important hyperparameter (i.e. the parameters that need to be chosen by the programmer before executing a machine learning program) that needs to be tuned .

If we choose a learning rate that is too small, the gradient descent algorithm might take a really long time to find the minimum value of the error function. This defeats the purpose of gradient descent, which was to use a computationally efficient method for finding the optimal solution.

On the other hand, if we choose a learning rate that is too large, we might overshoot the minimum value of the error function, and may even never reach the optimal solution. 
