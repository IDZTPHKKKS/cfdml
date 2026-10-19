class CfdmlError(Exception):
    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint
