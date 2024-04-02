import numpy as np
from scipy import sparse, linalg


class SoftImputeALS:
    def __init__(self, r, lambda_, min_value, max_value, max_iter, eval_fun):
        self.r = r
        self.lambda_ = lambda_
        self.min_value = min_value
        self.max_value = max_value
        self.max_iter = max_iter
        self.eval_fun = eval_fun

    def projection(self, X, A, B):
        I, J = X.nonzero()
        return sparse.coo_matrix(
            (np.sum(A[I, :] * B[J, :], axis=1), (I, J)),
            shape=X.shape,
        )
    
    def obj_func(self, X, A, B):
        return 0.5 * (
            sparse.linalg.norm(X - self.projection(X, A, B), ord="fro") +
            self.lambda_ * linalg.norm(A, ord="fro") +
            self.lambda_ * linalg.norm(B, ord="fro")
        )

    def fit(self, X):
        m, n = X.shape
        U, _ = linalg.qr(np.random.randn(m, self.r))
        U = U[:, :self.r]
        D = np.ones((self.r, 1))
        V = np.zeros((n, self.r))

        for i in range(1, self.max_iter + 1):
            # Update B
            A = U * D.T
            B = V * D.T
            P = self.projection(X, A, B)

            obj = self.obj_func(X, A, B)
            mae = self.eval_fun(X, P, X.count_nonzero())
            print(f"Iteration: {i:>3} | Objective: {obj:.4f} | MAE: {mae:.4f}")

            S = X - P
            shrink = D / (D ** 2 + self.lambda_)
            B = ((shrink * U.T) @ S).T + V * (D ** 2).T * shrink.T
            V, D, _ = linalg.svd(B * D.T, full_matrices=False)
            V = V[:, :self.r]
            D = np.expand_dims(D[:self.r] ** 0.5, axis=1)

            # Update A
            A = U * D.T
            B = V * D.T
            P = self.projection(X, A, B)
            S = X - P
            shrink = D / (D ** 2 + self.lambda_)
            A = S @ (V * shrink.T) + U * (D ** 2).T * shrink.T
            U, D, _ = linalg.svd(A * D.T, full_matrices=False)
            U = U[:, :self.r]
            D = np.expand_dims(D[:self.r] ** 0.5, axis=1)

        self.A = U * D.T
        self.B = V * D.T

    def predict(self, i, j):
        return int(np.clip(self.A[i, :] @ self.B[j, :].T, self.min_value, self.max_value))
