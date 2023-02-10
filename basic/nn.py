# import numpy as np
# import pandas as pd

# pd

# ### Stupidly applied gradient descent with a regression model
# # # Initialize weights and biases
# # w = np.random.rand(x.shape[1], 1)  # theta.shape = (784, 1)
# # b = np.random.rand()

# # Js = []
# # alpha = 5e-7
# # for i in tqdm(range(20)):
# #     y_pred = x @ w + b
# #     slope = 2 * np.mean((y_pred - y) * x, axis=0)
# #     w -= alpha * slope.reshape(-1, 1)
# #     b -= alpha * np.mean(y_pred - y)
    
# #     J = np.mean((y - y_pred) ** 2)
# #     Js.append(J)