import numpy as np

from rbmclass import BernoulliRBM_plus

class StackedRBMs(object):
    def __init__(self, layer_sizes, sample_activations=False):
        """
        - layer_sizes should be a sequence of integers, with the desired sizes
          of the visible layer and hidden layers (so it also defines the number
          of layers).
        - sample_activations is a boolean and determines whether we actually
          sample a binary value for h from its activation p(h|v) during
          transforming and training.
        """
        self.rbms = [
            BernoulliRBM_plus(n_components=h_size)
            for h_size in layer_sizes[1:]  # skip the visible layer
        ]

        self.sample_activations = sample_activations


    def transform(self, v):
        """Transform the given samples to obtain hidden unit activations

           Returns a list of activations [v, h1, h2, ...]. Activations can be
           seen as the probability of each hidden unit being on given the input,
           p(h|v). The inputs to a higher layer are the activations of the
           layer below it.
        """
        activations = [v]
        for rbm in self.rbms:
            rbm_input = activations[-1]
            activation = rbm.transform(rbm_input)
            if self.sample_activations:
                activation = 1 * (np.random.sample() < activation)
            activations.append(activation)
        return activations


    def train(self, *activations):
        """Train the model with given inputs

           For normal (unsupervised) training, pass a 2d array with each row
           being a training sample. Similar arrays with values for the hidden
           layers can be passed as subsequent arguments and are used to push
           the weights in a desired direction.
        """
        activations = list(activations)
        # Pad activations to length of layers
        n_missing = 1+len(self.rbms) - len(activations)
        activations += [None] * n_missing

        # Train layer by layer
        for i, rbm in enumerate(self.rbms):
            rbm_input = activations[i]
            if activations[i+1] is None:
                # Unsupervised
                rbm.partial_fit(rbm_input)
                activation = rbm.transform(rbm_input)
                if self.sample_activations:
                    activation = 1 * (np.random.sample() < activation)
                activations[i+1] = activation
            else:
                # 'Supervised'
                rbm.partial_fit(rbm_input, h=activations[i+1])


    @property
    def parameters(self):
        return [
            {
                'w': rbm.components_,
                'bias_h': rbm.intercept_hidden_,
                'bias_v': rbm.intercept_visible_,
            }
            for rbm in self.rbms
        ]
