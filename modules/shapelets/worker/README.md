# Shapelets Python Worker

This project contains the shapelets worker written in Python. The worker implements
the communication protocol with the worker manager, which is part of shapelets.

## Build

```bash
python -m pip install --upgrade pip
git submodule update --init
python setup.py generate_proto"
```