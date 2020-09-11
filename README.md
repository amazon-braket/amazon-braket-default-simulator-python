# Amazon Braket Default Simulator

[![Latest Version](https://img.shields.io/pypi/v/amazon-braket-default-simulator.svg)](https://pypi.python.org/pypi/amazon-braket-default-simulator)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/amazon-braket-default-simulator.svg)](https://pypi.python.org/pypi/amazon-braket-default-simulator)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/amazon-braket-default-simulator-python/badge/?version=latest)](https://amazon-braket-default-simulator-python.readthedocs.io/en/latest/?badge=latest)

The Amazon Braket Default Simulator is a Python open source library that provides an implementation of a quantum simulator 
that you can run locally. You can use the simulator to test quantum tasks that you construct for the [Amazon Braket SDK](https://github.com/aws/amazon-braket-sdk-python)
before you submit them to the Amazon Braket service for execution.

## Setting up Amazon Braket Default Simulator Python
You must have the [Amazon Braket SDK](https://github.com/aws/amazon-braket-sdk-python) installed to use the local simulator.
Follow the instructions in the [README](https://github.com/aws/amazon-braket-sdk-python/blob/main/README.md) for setup.

**Checking the version of the DefaultSimulator**

You can check your currently installed version of `amazon-braket-default-simulator` with `pip show`:

```bash
pip show amazon-braket-default-simulator
```

or alternatively from within Python:

```
>>> from braket import default_simulator
>>> default_simulator.__version__
```

## Usage
The quantum simulator implementation `DefaultSimulator` plugs into the `LocalSimulator` interface in 
[Amazon Braket SDK](https://github.com/aws/amazon-braket-sdk-python) using the `backend` value as `"default"`. 

**Executing a circuit using the DefaultSimulator**
```python
from braket.circuits import Circuit
from braket.devices import LocalSimulator

device = LocalSimulator("default")

bell = Circuit().h(0).cnot(0, 1)
print(device.run(bell, shots=100).result().measurement_counts)
```

## Documentation

Detailed documentation, including the API reference, can be found on [Read the Docs](https://amazon-braket-default-simulator-python.readthedocs.io/en/latest/)

**To generate the API Reference HTML in your local environment**

First, install tox:

```bash
pip install tox
```

To generate the HTML, first change directories (`cd`) to position the cursor in the `amazon-braket-default-simulator-python` directory. Then, run the following command to generate the HTML documentation files:

```bash
tox -e docs
```

To view the generated documentation, open the following file in a browser:
`../amazon-braket-default-simulator-python/build/documentation/html/index.html`

## Testing

If you want to contribute to the project, be sure to run unit tests and get a successful result 
before you submit a pull request. To run the unit tests, first install the test dependencies using the following command:
```bash
pip install -e "amazon-braket-default-simulator-python[test]"
```

To run the unit tests:
```bash
tox -e unit-tests
```

To run the performance tests:
```bash
tox -e performance-tests
```
These tests will compare the performance of a series of simulator executions for your changes against the latest commit on the main branch.
*Note*: The execution times for the performance tests are affected by the other processes running on the system.
In order to get stable results, stop other applications when running these tests.

To run selected tests based on a key word in the test name:
```bash
tox -- -k 'keyword'
```


## License

This project is licensed under the Apache-2.0 License.

