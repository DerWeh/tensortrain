# TensorTrain
Tensor methods (DMRG and TDVP) using the tensor-train geometry.
This is the code supplementing our computational physics lecture.
The code is not meant to be efficient and should only be used for didactic purposes.

We used the [TensorNetwork](https://github.com/google/tensornetwork) library as framework for the tensor networks.
If you want to write performant code, have a look at [ITensor](http://itensor.org/).
An extensive Python library is [QUIMP](https://quimb.readthedocs.io/en/latest/).


The directory `notebooks` contains the tutorials:
* `01_basic_tensor_operations.ipynb`  
  Basic introduction to tensor networks and the necessary operations.
* `01_basic_tensor_operations_tn.ipynb`  
  Usage of the [TensorNetwork](https://github.com/google/tensornetwork) library to perform those operations more conveniently.
* `02_dmrg.ipynb`  
  Step-by-step instructions to perform a DMRG sweep.

`tensortrain` is the Python module to import.
The scripts `siam.py` and `heisenberg_xx.py` are two example scripts that can be run.

# Installation
Install the dependencies in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
