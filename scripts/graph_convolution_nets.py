import numpy as np

# nodes
nodes = [f"N{i}" for i in range(0, 5)]

# labels
num_classes = 2
Y = np.eye(num_classes)[[0,0,0,1,1]]

# input
X = np.array([[1,0,2], [0,1,1], [2,1,0], [1,2,1], [0, 1, 0]], dtype = float)

# edges
edges = [(0,1), (1,2), (2,3), (3,4), (0,2)]

N = len(nodes)

# adjacency matrix
A = np.zeros((N,N), dtype = float)
for i,j in edges:
    A[i,j] = 1
    A[j,i] = 1

# identity matrix
I = np.eye(N, dtype = float)

# add self-loops to adjacency matrix
A_tilde = A + I

# degree per node
deg = np.sum(A_tilde, axis = 1)
# degree matrix
D_tilde = np.diag(deg)
# D^(-1/2)
D_tilde_inv_sqrt = np.diag(1.0/np.sqrt(deg))
# A_hat
A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt # normalized adjacency matrix 

# why normalized adjacency matrix? Because without normalization, the node with high degree will have larger value after aggregation, which may lead to numerical instability. This means feature magnitude will depend heavily on node degree than the structure of the graph.
# normalized adjacency matrix balances the influence of nodes with high degrees; keeps symmetry; Avoid exploding/vanishing gradients

# Neighborhood mixing

S0 = A_hat @ X

hidden_dim = 4

W0 = np.random.randn(3,4)

# Pre-activation

H1_pre = S0 @ W0

# Activation

H1 = np.maximum(H1_pre, 0) # ReLU

# Neighborhood mixing again

S1 = A_hat @ H1


W1 = np.random.randn(4,2)

# Pre-activation

H2_pre = S1 @ W1 # logits

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

H2 = softmax(H2_pre) # probabilities

predictions = np.argmax(H2, axis=1)


eps = 1e-12
if 'Y' in globals():
    loss = -(Y * np.log(np.clip(np.asarray(predictions).reshape(5,1), eps, 1.0))).sum(axis=1).mean()
    print("Cross-entropy loss:", float(loss))



# --- Prereqs expected: X (n x d_in), A_hat (n x n), Y (n x C one-hot) ---
n, d_in = X.shape
num_classes = Y.shape[1]
hidden_dim = 4

# --- Helpers ---
def relu(M): return np.maximum(0.0, M)
def relu_grad(M): return (M > 0).astype(M.dtype)
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

# --- Init parameters ---
rng = np.random.default_rng(0)
W0 = rng.normal(0, 0.3, size=(d_in, hidden_dim))          # (d_in x hidden)
W1 = rng.normal(0, 0.3, size=(hidden_dim, num_classes))    # (hidden x C)

# --- Training hyperparams ---
lr = 0.2
steps = 10000
loss_history = []

for step in range(1, steps + 1):
    # 1) FORWARD -------------------------------------------------------------
    S0      = A_hat @ X                 # (n x d_in)   neighbor mix
    H1_pre  = S0 @ W0                   # (n x hidden) linear
    H1      = relu(H1_pre)              # (n x hidden) ReLU
    S1      = A_hat @ H1                # (n x hidden) second mix
    logits  = S1 @ W1                   # (n x C)
    probs   = softmax(logits)           # (n x C)

    # 2) LOSS ---------------------------------------------------------------
    eps  = 1e-12
    loss = -(Y * np.log(np.clip(probs, eps, 1.0))).sum(axis=1).mean()

    # 3) TRACK LOSS ---------------------------------------------------------
    loss_history.append(loss)
    if step % 10 == 0 or step == 1:
        print(f"step {step:3d} | loss {loss:.4f}")

    # 4) GRADIENTS (BACKPROP) ----------------------------------------------
    # dL/dlogits for softmax + CE over all n samples
    dlogits = (probs - Y) / n                           # (n x C)

    # Grad W1 and backprop to S1
    dW1 = S1.T @ dlogits                                # (hidden x C)
    dS1 = dlogits @ W1.T                                # (n x hidden)

    # Back through S1 = A_hat H1  (A_hat is symmetric)
    dH1 = A_hat @ dS1                                   # (n x hidden)

    # Back through ReLU
    dH1_pre = dH1 * relu_grad(H1_pre)                   # (n x hidden)

    # Grad W0
    dW0 = S0.T @ dH1_pre                                # (d_in x hidden)

    # 5) UPDATE PARAMETERS --------------------------------------------------
    W1 -= lr * dW1
    W0 -= lr * dW0

# --- Final evaluation ---
preds = probs.argmax(axis=1)
acc = (preds == Y.argmax(axis=1)).mean()
print("final acc:", acc, "| preds:", preds)









