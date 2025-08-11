# Overview

This guide explains how to implement a new attack to marked object. For more examples, refer to the `wibench.attacks` module.

## 0. Implement it as a plugin

Create `your_attack.py` file in `user_plugins` directory

## Custom attack

Attack class should inherit `BaseAttack` class and implement `__call__` method.

```python
from wibench.attacks import BaseAttack


class MyAttack(BaseAttack):
    def __init__(self, any_parameters_of_atack):
        ...

    def __call__(self, object_to_attack):
        # Attack input object here
        ...
        return attacked_object
