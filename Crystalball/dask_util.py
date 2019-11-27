class Key(object):
    """ Lightweight key object, small enough to pass for a dask collection """
    __slots__ = ("key",)
    DUMMY_GRAPH = {}

    def __init__(self, key):
        self.key = key

    def __dask_graph__(self):
        return self.DUMMY_GRAPH

    def __dask_keys__(self):
        return (self.key,)

    def __repr__(self):
        return str(self.key)

    __str__ = __repr__
