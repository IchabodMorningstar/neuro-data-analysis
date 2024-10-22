from pickle import dump, load
import os
import sqlite3
from contextlib import closing
from uuid import uuid4
import json

class Step:
    def __init__(self, func, parents=None, **kwargs):
        self.name = func.__name__.lstrip('_')
        if isinstance(parents, self.__class__):
            self.parents = [parents]
        elif parents is None:
            self.parents = []
        elif isinstance(parents, list):
            self.parents = parents
        else:
            raise ValueError("parents must be a list of steps or a single step")
        self.kwargs = kwargs
        self.func = func
        self.value = None

    def _format_kwargs(self):
        return '_'.join([f"{k}={v}" for (k, v) in self.kwargs.items()])

    def key(self):
        # if self.parents is None:
        #     return {'_name': self.name, **self.kwargs}
            # return f"{self.name}({self._format_kwargs()})"
        # paths = [f"{self.prev_step.path()}/{self.name}({self._format_kwargs()})" for self.prev_step in self.parents]
        # return ''.join(paths)
        return {'_name': self.name, **self.kwargs, 
                "_parents": [prev_step.key() for prev_step in self.parents]}
    
    def __call__(self, ignore_cache=False):
        if (self.value): return self.value

        # print(json.dumps(self.key()))

        def insert(cursor, file):
            cursor.execute("INSERT INTO cache (key, file) VALUES (?, ?)", (json.dumps(self.key()), file))

        with closing(sqlite3.connect("../data/processed/index.db", autocommit=True)) as connection:
            with closing(connection.cursor()) as cursor:
                try:
                    file = cursor.execute("SELECT file FROM cache WHERE key = ?", [json.dumps(self.key())]).fetchone()
                    if not file:
                        file = uuid4().hex
                        insert(cursor, file)
                    else: file = file[0]
                except sqlite3.OperationalError:
                    cursor.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, file TEXT)")
                    file = uuid4().hex
                    insert(cursor, file)
                
        cache_path = f"../data/processed/{file}.pkl"
        # input = self.prev_step() if self.prev_step else {}
        input = {}
        if self.parents is not None:
            for step in self.parents:
                input.update(step())

        if not ignore_cache:
            try:
                with open(cache_path, 'rb') as f:
                    d = load(f)
                    # print(f"{self.path()} found in cache")
                    input.update(d)
                    return input
            except FileNotFoundError: pass

        output = self._run(input)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        dump(output, open(cache_path, 'wb'))
        input.update(output)
        self.value = input
        return input

    def _run(self, input):
        # print(f"running {self.name} with {self.kwargs} and {list(input.keys())}")
        return self.func(input, **self.kwargs)