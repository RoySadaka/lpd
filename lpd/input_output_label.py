class InputOutputLabel():
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.labels = None

    def update(self, inputs, outputs, labels):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels