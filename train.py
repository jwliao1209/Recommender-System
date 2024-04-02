import numpy as np
from scipy import sparse

from src.soft_impute_als import SoftImputeALS
from src.metric import cal_mae, cal_rmse


def split_train_test(data, test_ratio=0.2):
    n = data.shape[0]
    test_index = np.random.randint(0, n, size=int(n * test_ratio))
    test_data = data[test_index, :]
    train_data = np.delete(data, test_index, axis=0)
    return train_data, test_data


def convert_to_sparse_matrix(data, m, n):
    users = data[:, 0].astype(int)
    movies = data[:, 1].astype(int)
    ratings = data[:, 2]
    return sparse.coo_matrix((ratings, (users - 1, movies - 1)), shape=(m, n))


if __name__ == "__main__":
    data = np.genfromtxt("data/movie.data")
    m, n = int(np.max(data[:, 0])), int(np.max(data[:, 1]))
    max_value, min_value = int(np.max(data[:, 2])), int(np.min(data[:, 2]))
    train_data, test_data = split_train_test(data)
    X_train = convert_to_sparse_matrix(train_data, m, n)
    model = SoftImputeALS(
        r=250,
        lambda_=0.1,
        max_value=max_value,
        min_value=min_value,
        max_iter=100,
        eval_fun=cal_mae,
    )
    model.fit(X_train)

    rating_list = []
    pred_list = []
    for i, j, r, _ in test_data:
        pred = model.predict(int(i) - 1, int(j) - 1)
        pred_list.append(pred)
        rating_list.append(r)
        print(f"(User, Movie): ({int(i+1):>3}, {int(j+1):>4}), Predict: {pred}, Rating: {int(r)}")
    
    ratings = np.array(rating_list)
    preds = np.array(pred_list)
    num = len(np.array(pred_list))
    print(f"test MAE: {cal_mae(pred, ratings, num):.4f}")
