## Contributing to Shapelets Compute

## Raise an Issue

Raising **[issues](https://github.com/shapelets/shapelets-compute/issues)** is welcome.

## Clone the Shapelets Compute

Prerequisites can be found [here](https://github.com/shapelets/shapelets-compute/blob/master/doc/user/installguide.rst#installation-from-source).  

```commandline
git clone git@github.com:shapelets/shapelets-compute.git
cd shapelets-compute
git submodule update --init --recursive
pip install -r requirements.txt
pip install -r requirements-doc.txt
pip install -r requirements-test.txt
python setup.py clean
python setup.py develop
```

## Contribute a PR

## Repository overview

- [demos](https://github.com/shapelets/shapelets-compute/tree/master/doc)
- [doc](https://github.com/shapelets/shapelets-compute/tree/master/doc)
  
  Contains the project documentation, ready to be built with 
  [Sphinx](https://www.sphinx-doc.org/en/master/):
  ```commandline 
  cd doc 
  pip install -r requirements-doc.tx
  sphinx-build . build  
  ```
- [etc](https://github.com/shapelets/shapelets-compute/tree/master/etc)  
- [modules/benchmarks](https://github.com/shapelets/shapelets-compute/tree/master/modules/benchmarks)
- [modules/gauss](https://github.com/shapelets/shapelets-compute/tree/master/modules/gauss)
- [modules/pygauss](https://github.com/shapelets/shapelets-compute/tree/master/modules/pygauss)
- [modules/shapelets](https://github.com/shapelets/shapelets-compute/tree/master/modules/shapelets)
- [modules/test](https://github.com/shapelets/shapelets-compute/tree/master/modules/test)


# Squashing commits

Squash your commits before we merge, it keeps our repository concise and clean.

- Make sure your branch is up to date with the master branch.
- Run `git rebase -i master`.
- You should see a list of picks.
- The first commit will remain "pick" and the rest become "f".
- Save and close the editor.
- Then force push the commit: `git push -f`.
