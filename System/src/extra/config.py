'''
From Mozilla Deep Speech, slightly modified to fit our project.
'''
from __future__ import absolute_import, division, print_function

import os
import sys

from attrdict import AttrDict

from tensorflow import flags

FLAGS = flags.FLAGS


from extra.text import Alphabet

class ConfigSingleton:
    _config = None

    def __getattr__(self, name):
        if not ConfigSingleton._config:
            raise RuntimeError("Global configuration not yet initialized.")
        if not hasattr(ConfigSingleton._config, name):
            raise RuntimeError("Configuration option {} not found in config.".format(name))
        return ConfigSingleton._config[name]


Config = ConfigSingleton() # pylint: disable=invalid-name

def initialize_globals():
    c = AttrDict()
    c.alphabet = Alphabet(os.path.abspath("dict/EN_space_chars.txt"))

    # Units in the sixth layer = number of characters in the target language plus one
    c.n_hidden_6 = c.alphabet.size() + 1 # +1 for CTC blank label
    ConfigSingleton._config = c # pylint: disable=protected-access