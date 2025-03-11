# TensorID

Public code for tensor interpolative decomposition (ID). This code provides reference implementations of the algorithms discussed in the following preprint:

> Zhang, Yifan, Fornace, Mark, and Lindsey, Michael. "Fast and Accurate Interpolative Decompositions for General and Structured Tensors." arXiv preprint (in prep, 2025)

Please cite this paper when using or referencing this software.

The main top-level functions are described below. See docstrings for full explanations of usage and `demo_*.py` for example calculations.

## CoreID

```python
CPSolver(...) # Initialize the solver with a tensor
CPSolver.fit(...) # Return the index sets
```

```python
CPSolver(...)
CPSolver.fit(...)
```

## SatID

```python
CPSolver(...)
CPSolver.fit(...)
```

```python
CPSolver(...)
CPSolver.fit(...)
```

## Copyright Notice

TensorID Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
