class InputOutputLabel():
    def __init__(self):
        self.inputs = None
        self.output = None
        self.labels = None

    def update(self, inputs, output, labels):
        self.inputs = inputs
        self.output = output
        self.labels = labels