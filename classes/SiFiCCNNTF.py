class SiFiCCNNTF:

    def __init__(self, model):
        # Tensorflow model
        self.model = model

    def train(self,x_train, y_train, x_valid, y_valid):