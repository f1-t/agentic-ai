"""Small key -> object registry used to select simulator components from config."""


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._items = {}

    def register(self, key):
        def decorator(obj):
            if key in self._items:
                raise ValueError(f"{self.kind} '{key}' is already registered")
            self._items[key] = obj
            return obj
        return decorator

    def get(self, key):
        if key not in self._items:
            raise KeyError(f"unknown {self.kind} '{key}', available: {self.keys()}")
        return self._items[key]

    def keys(self):
        return sorted(self._items)

    def __contains__(self, key):
        return key in self._items
