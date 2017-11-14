import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import load_iris, fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


class Input:
    """Input layer, only passes data forward"""

    def __init__(self, input_size):
        if input_size <= 0:
            raise ValueError("Input must have at least one value.")
        self.input_size = input_size
        self.output_size = input_size
        self.activation = np.zeros((input_size, 1), dtype=np.float32)

    def forward(self, values):
        if len(values) != self.input_size:
            raise ValueError("Input layer expected {} values, {} received.".format(values, self.input_size))
        self.activation = values
        return values

    def reset(self):
        pass  # Nothin to do, probably should implement some kind of base class at this point


class FullyConnected:
    def __init__(self, input_size, n_neurons):
        self.input_size = input_size
        self.output_size = n_neurons
        self.z = np.zeros((n_neurons, 1), dtype=np.float32)  # Neuron output
        self.activation = np.zeros((n_neurons, 1), dtype=np.float32)  # Neuron output with activation func applied
        self.w = np.random.randn(n_neurons, input_size)
        self.bias = np.random.randn(n_neurons, 1)

    def forward(self, values):
        if len(values) != self.input_size:
            raise ValueError("fc layer expected {} values, {} received.".format(values, self.input_size))
        self.z = np.dot(self.w, values) + self.bias
        self.activation = sigmoid(self.z)
        return self.activation

    def reset(self):
        self.w = np.random.randn(self.output_size, self.input_size)
        self.bias = np.random.randn(self.output_size, 1)


class MyMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.005, hidden_layer_sizes=[], n_epochs=100):
        """If params is None the method is initialized with default values.
           n_neurons is an array with size hidden_layers that defiens how many
           neurons each hidden layer will have."""
        self.lr = lr
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_epochs = n_epochs

    def predict(self, X):
        try:
            getattr(self, "layers_")
        except AttributeError:
            raise RuntimeError("You must train the classifier before predicting!")

        if X.ndim == 1:
            if len(X) != self.input_size_:
                raise ValueError(
                    f"Invalid argument passed to predict: X must have {self.input_size_} values, {len(X)} passed"
                )

            result = X[:, np.newaxis]  # This turns a (1, x) array into a (x, 1) ndarray

            for layer in self.layers_:
                result = layer.forward(result)
            return result

        elif X.ndim == 2:
            if X.shape[1] != self.input_size_:
                raise ValueError(
                    f"Invalid argument passed do predict: X must be of shape (n_samples, {self.input_size_}), {X.shape} passed."
                )

            result_dim = (X.shape[0], self.output_size_, 1)
            predictions = np.zeros(result_dim, dtype=np.float32)

            for i, to_predict in enumerate(X):
                predictions[i] = self.predict(to_predict)
            return predictions

        else:
            raise ValueError("Invalid argument passed to predict: Too many dimensions")

    def score(self, X, y, sample_weight=None):
        y = y[:, :, np.newaxis]
        correct_predictions = 0
        for m, prediction in enumerate(self.predict(X)):
            if np.argmax(prediction) == np.argmax(y[m]):
                correct_predictions += 1
        print(correct_predictions / len(X))
        return correct_predictions / len(X)

    def reset(self):
        for layer in self.layers_:
            layer.reset()

    def fit(self, X, y, keep_weights=False, use_tqdm=False):
        already_trained = True

        try:
            getattr(self, "layers_")
        except AttributeError:
            already_trained = False

        if not (already_trained and keep_weights):
            self.loss_history_ = []
            self.layers_ = []
            self.input_size_ = X.shape[1]
            self.output_size_ = y.shape[1]
            self.layers_.append(Input(self.input_size_))

            for i, neurons in enumerate(self.hidden_layer_sizes):
                self.layers_.append(
                    FullyConnected(input_size=self.layers_[i].output_size, n_neurons=neurons)
                )

            self.layers_.append(
                FullyConnected(input_size=self.layers_[-1].output_size, n_neurons=self.output_size_)
            )

        y = y[:, :, np.newaxis]

        b_gradients = np.array([np.zeros(layer.bias.shape) for layer in self.layers_[1:]])
        w_gradients = np.array([np.zeros(layer.w.shape) for layer in self.layers_[1:]])

        for k in range(self.n_epochs):
            total_error = 0

            p_bar = enumerate(zip(X, y))

            if use_tqdm:
                p_bar = tqdm(p_bar, desc="Epoch {}".format(k))

            for i, (input_values, expected) in p_bar:
                prediction = self.predict(input_values)  # Predict returns the activations of the last layer

                total_error += mean_squared_error(expected[:, 0], prediction[:, 0])

                # Calculate the delta  of the last layer
                delta = (prediction - expected) * dsigmoid(prediction)

                b_gradients[-1] = delta  # Gradient of bias is the error
                w_gradients[-1] = np.dot(delta, self.layers_[-2].activation.T)

                # Calculate the error and gradient of all the other layers
                for j in range(2, len(self.layers_)):
                    next_delta = b_gradients[-j+1]  # delta of the next layer is already stored as the gradient of bias
                    cur_delta = np.dot(self.layers_[-j+1].w.T, next_delta) * dsigmoid(self.layers_[-j].activation)
                    b_gradients[-j] = cur_delta
                    w_gradients[-j] = np.dot(cur_delta, self.layers_[-j - 1].activation.T)

                # Finally, update the weights
                for l in range(1, len(self.layers_)):
                    self.layers_[-l].bias -= (b_gradients[-l] * self.lr)
                    self.layers_[-l].w -= (w_gradients[-l] * self.lr)

            average_loss = total_error / len(X)
            #print("Epoch {} average loss: {}".format(k, average_loss))
            self.loss_history_.append(average_loss)

            b_gradients.fill(0)
            w_gradients.fill(0)
        return self


# Remember to set the class name appropiately.

def distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))


class KNNClassifier(BaseEstimator, ClassifierMixin):  # or RegressonMixin?
    def __init__(self, k=3, weighted=False):
        self.weighted = weighted
        self.k = k

    def score(self, X, y, sample_weight=None):
        predictions = self.predict(X)
        return np.sum(y == predictions) / len(predictions)

    def predict(self, X):
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1]:
            raise ValueError(
                "Invalid arguments passed to predict: Expected array with dimensions (n_samples, n_dim), got {}"
                .format(X.shape)
            )

        try:
            getattr(self, "X_")
        except AttributeError:
            raise RuntimeError("You must train the classifier before predicting!")

        predictions = np.zeros((X.shape[0]), dtype=np.int32)

        for i, sample in enumerate(X):
            distances = np.zeros(len(self.X_), dtype=np.float32)
            for j, elem in enumerate(self.X_):
                distances[j] = distance(sample, elem)

            indexes = np.argsort(distances)

            closest = self.y_[indexes[:self.k]]

            if self.weighted:
                distances_of_closest = distances[indexes[:self.k]]
                n_outputs = len(np.unique(self.y_))
                results = np.zeros(n_outputs, dtype=np.float32)
                for p in range(n_outputs):
                    which = np.where(closest == p)
                    weighted_distances = np.divide(1, distances_of_closest[which], where=distances_of_closest[which] != 0)
                    results[p] = np.sum(weighted_distances)
                    # results[i] = np.sum(1/distances_of_closest[np.where(closest == p)])
                predictions[i] = np.argmax(results)
            else:
                freq = np.unique(closest, return_counts=True)
                predictions[i] = freq[0][0]

        return predictions

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y

        return self


def test_mlp_on_mnist():
    mnist_data = fetch_mldata("MNIST original")

    encoded_mnist_y = OneHotEncoder(dtype=np.float32, sparse=False).fit_transform(mnist_data.target.reshape(-1, 1))

    X, y, unencoded_y = resample(mnist_data.data, encoded_mnist_y, mnist_data.target, random_state=0, n_samples=70000)

    X = np.divide(X, 255)  # Scale to 0-1

    classifier = MyMLP(hidden_layer_sizes=[64, 32])

    classifier.fit(X, y)

    classifier.score(X, y)

def test_knn():
    iris_data = load_iris()
    model = KNNClassifier()
    model.fit(iris_data.data, iris_data.target)
    model.score(iris_data.data, iris_data.target)

    params = {"k": [1, 3, 5, 10], "weighted": [True, False]}
    grid = GridSearchCV(KNNClassifier(), param_grid=params, cv=3)
    grid.fit(iris_data.data, iris_data.target)
    print(grid.best_params_, grid.best_score_)


if __name__ == "__main__":
    test_knn()
    test_mlp_on_mnist()




