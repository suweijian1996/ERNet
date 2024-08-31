import numpy as np

scales =  np.array([128, 256, 512])
aspect_ratios = np.array([0.5, 1, 2])

h_ratios = np.sqrt(aspect_ratios)
w_ratios = 1 / h_ratios
print(h_ratios)
print(w_ratios)
print(h_ratios[:, None])
print(w_ratios[:, None])

ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)
print(ws, '\n',hs)