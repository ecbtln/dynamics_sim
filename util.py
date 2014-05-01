__author__ = 'elubin'


class Obj:
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
