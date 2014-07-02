Team Konnectomics
==============================================================================
Python code used for the Kaggle Connectomics Challenge

***

Installation
-------------

Install the dependencies listed in `DEPENDENCIES` (we recommend using a separate `virtualenv`)

Then, from the root of the source tree, call

    $ python setup.py build_ext --inplace

... and that's it


Useage
-------

Our code assumes that the evaluation datasets are stored in PyTables HDF5 files, with each dataset comprising a group containing three arrays named `fluorescence`, `network` and `network_pos`, consisting of the simulated flourescence data, the true set of connections, and the xy positions of the neurons respectively.

The main script used to process an evaluation dataset is `basic_workflow.run()`. To construct a submission from a pair of `validation` and `test` datasets, use `basic_workflow.make_submission()`.

Most of the important core functions are well documented. If you have any other questions, please email alistair.muldal[at]pharm.ox.ac.uk

Copyright 2014 Alistair Muldal