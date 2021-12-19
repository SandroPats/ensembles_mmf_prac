class TargetError(ValueError):
    def __init__(self, message):
        super().__init__(message)


class DatasetError(ValueError):
    def __init__(self, message):
        super().__init__(message)
