class CallbackMonitorResult():
    def __init__(self, did_improve: bool,
                        new_value: float,
                        prev_value: float,
                        new_best: float,
                        prev_best: float,
                        change_from_previous: float,
                        change_from_best: float,
                        patience_left: int,
                        description: str,
                        name: str):
        self.name = name
        self.did_improve = did_improve
        self.new_value = new_value
        self.prev_value = prev_value
        self.new_best = new_best
        self.prev_best = prev_best
        self.change_from_previous = change_from_previous
        self.change_from_best = change_from_best
        self.patience_left = patience_left
        self.description = description

    def has_improved(self):
        return self.did_improve

    def has_patience(self):
        return self.patience_left > 0
