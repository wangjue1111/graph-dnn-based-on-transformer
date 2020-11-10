import numpy as np

def get_laplacian(matrix):
    matrix=matrix.squeeze(0)
    D=np.diag(matrix.sum(0))
    D12=np.linalg.inv(np.sqrt(D))
    a=np.matmul(D12,matrix)
    return np.matmul(a,D12)

a=np.eye(5)
print(np.linalg.inv(np.sqrt(a)))