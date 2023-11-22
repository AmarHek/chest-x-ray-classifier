from dataclasses import dataclass, asdict
from pprint import pprint
from typing import Dict, List


@dataclass
class BaseParams:
    name: str = "BaseParams"

    def print_params(self):
        attributes = self.__dict__
        print("--------------------------------------")
        print(self.name.upper())
        print("--------------------------------------")
        for key, value in attributes.items():
            print("{:<20} {:<20}".format(key, str(value)))
        print("--------------------------------------")

    def pprint_params(self):
        pprint(self)

    def load_from_dict(self, dictionary: Dict):
        self.assert_keys(list(dictionary.keys()))
        for key, value in dictionary.items():
            target_type = type(getattr(self, key))
            value = target_type(value)
            setattr(self, key, value)

    def assert_keys(self, keys: List):
        unexpectedKeys = []
        for key in keys:
            if key not in self.__dict__.keys():
                unexpectedKeys.append(key)
        if len(unexpectedKeys) > 0:
            unexpectedKeys = ', '.join(unexpectedKeys)
            raise KeyError("unexpected key(s): %s" % unexpectedKeys)

    def to_dict(self):
        # convert contents of dataclass to dict
        return asdict(self)
