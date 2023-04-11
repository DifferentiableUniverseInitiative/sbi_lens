# Utility scripts


## Computing Log Normal Shift Parameters with Cosmomentum

We provide the script `compute_log_normal_shifts.py` to evaluate the shift parameters on a range of cosmological parameter values
and for a baseline set of redshift distributions.

To use it, make sure to first compile Cosmomentum as follows:
```bash
cd third_party/CosMomentum/cpp_code; make DSS
```
This should create the DSS.so library that we link to in this folder.

Then you can run the script as:
```bash
python compute_log_normal_shifts.py
```
It will use multiprocessing to distribute the computation on as many CPUs as you have available on your machine.
