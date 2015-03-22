import numpy as np
import theano

from mlp_theano import MLP_theano, train_mlp
from logistic_sgd import logreg, shared_dataset


class MLP(object):
    def __init__(self, layer_sizes):
        self.n_layers = len(layer_sizes)

        classifier = MLP_theano(
            rng = np.random.RandomState(),
            n_in=layer_sizes[0],
            n_hidden=layer_sizes[1],
            n_out=layer_sizes[-1]
        )

        x = classifier.x
        self.transform_func = theano.function(
            inputs=[x],
            outputs=[
                classifier.hiddenLayer.output,
                classifier.logRegressionLayer.p_y_given_x,
            ],
        )

        self.classifier = classifier


    def transform(self, v):
        h, y = self.transform_func(v)

        return [v, h, y]


    def train(self, *activations):
        if activations[1] is None:
            # No hidden layer activations specified, do normal supervised training
            self.normal_supervised_training(activations[0], activations[2])
        else:
            self.fully_supervised_training(*activations)


    @property
    def parameters(self):
        return [
            {
                'w': self.classifier.hiddenLayer.W.get_value().T,
                'b': self.classifier.hiddenLayer.b.get_value(),
            },
            {
                'w': self.classifier.logRegressionLayer.W.get_value().T,
                'b': self.classifier.logRegressionLayer.b.get_value(),
            }
        ]


    def normal_supervised_training(self, x, y):
        # transform one-of-n representation to int labels
        y = np.argmax(y, axis=1)

        # Split training set into training and validation set
        n_train = len(x)//2
        training_set = shared_dataset(x[:n_train],
                                      y[:n_train])
        validation_set = shared_dataset(x[n_train:],
                                        y[n_train:])
        train_mlp(self.classifier, training_set, validation_set)


    def fully_supervised_training(self, x, h, y):
        self.train_bottom(x, h)
        self.train_top(h, y)


    def train_bottom(self, x, h):
        """Train the hidden layer units to predict desired labels.

           Each hidden unit is trained with logistic regression. The resulting
           weights are inserted into the MLP's hidden layer.
        """

        # Split training set into training and validation set
        n_train = len(x)//2
        x_train = x[:n_train]
        h_train = h[:n_train]
        x_valid = x[n_train:]
        h_valid = h[n_train:]

        # Create matrices for weights and biases (values will be overwritten)
        w = self.classifier.hiddenLayer.W.get_value()
        b = self.classifier.hiddenLayer.b.get_value()

        n_hidden_features = h_train.shape[1]
        for unit in range(n_hidden_features):
            # Train this particular hidden unit
            target_train = h_train[:,unit]
            target_valid = h_valid[:,unit]
            training_set = shared_dataset(x_train, target_train)
            validation_set = shared_dataset(x_valid, target_valid)

            n_classes = 2
            n_inputs = x_train.shape[1]
            h_predictor = logreg(training_set, validation_set,
                                 n_inputs, n_classes)
            w[:,unit] = h_predictor.W.get_value()[:,1].T
            b[unit] = h_predictor.b.get_value()[1]

        # Insert the obtained weights into the MLP
        self.classifier.hiddenLayer.W.set_value(w)
        self.classifier.hiddenLayer.b.set_value(b)


    def train_top(self, h, y):
        """Train the logistic regression output layer separately"""
        # transform one-of-n representation to int labels
        y = np.argmax(y, axis=1)

        # Split training set into training and validation set
        n_train = len(h)//2
        h_train = h[:n_train]
        y_train = y[:n_train]
        h_valid = h[n_train:]
        y_valid = y[n_train:]

        # Train the logistic regressor on top
        training_set = shared_dataset(h_train, y_train)
        validation_set = shared_dataset(h_valid, y_valid)

        # XXX: this boldly assumes highest label appears at least once
        n_classes = 1+np.max(y_train)

        n_inputs = h_train.shape[1]
        y_predictor = logreg(training_set, validation_set,
                             n_inputs, n_classes)
        W = y_predictor.W.get_value()
        b = y_predictor.b.get_value()
        # Manually insert the values into the MLP top layer
        self.classifier.logRegressionLayer.W.set_value(W)
        self.classifier.logRegressionLayer.b.set_value(b)
