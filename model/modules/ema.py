"""
Exponential Moving Average for model parameters.
"""
import sys

class EMA():

    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        decay = min(self.mu, (1.0 + num_updates) / (10.0 + num_updates))
        # print(self.shadow)
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                name = name.replace("_layers.","")
                # print(name)
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        # sys.exit(0)

    def assign(self, model):
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                assert name in self.shadow
                self.original[name] = param.clone()
                param = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                assert name in self.shadow
                param = self.original[name]
