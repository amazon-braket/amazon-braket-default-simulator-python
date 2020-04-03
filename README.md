**This prerelease documentation is confidential and is provided under the terms of your nondisclosure agreement with Amazon Web Services (AWS) or other agreement governing your receipt of AWS confidential information.**

Amazon Braket Default Simulator Python is an open source library that provides an implementation of a quantum simulator 
which you can execute locally. This implementation can be used to test small instances of quantum tasks constructed 
using the [Amazon Braket SDK](https://github.com/aws/braket-python-sdk), before you submit them to the Amazon Braket 
service for execution. 

## Prerequisites

### Python 3.7.2 or greater
Download and install Python 3.7.2 or greater from [Python.org](https://www.python.org/downloads/).
If you are using Windows, choose **Add Python to environment variables** before you begin the installation.

### Amazon Braket SDK
Download and install the Amazon Braket SDK. Follow the instructions in the [README](https://github.com/aws/braket-python-sdk/blob/stable/latest/README.md).

## Setting up Amazon Braket Default Simulator Python
After you install the Amazon Braket SDK, either clone or download the amazon-braket-default-simulator-python repo to your local environment. 
You must clone or download the repo into a folder in the same virtual environment where you are using the Amazon Braket SDK.

Use the following command to clone the repo.

```bash
git clone https://github.com/aws/amazon-braket-default-simulator-python.git
```

Note that you must have a valid SSH key created in your local environment that has been added to your GitHub account to clone the repo.

You can also download the repo as a .zip file by using the **Clone or download** button. 

After you add the repo to your local environment, install the plugin with the following `pip` command:

```bash
pip install -e amazon-braket-default-simulator-python
```

## Usage
The quantum simulator implementation DefaultSimulator plugs into the `LocalSimulator` interface in 
the [Amazon Braket SDK](https://github.com/aws/braket-python-sdk) using the `backend` value as `"default"`.

**Executing a circuit using the DefaultSimulator**
```python
from braket.circuits import Circuit
from braket.devices import LocalSimulator

device = LocalSimulator("default")

bell = Circuit().h(0).cnot(0, 1)
print(device.run(bell, shots=1000).result.measurement_counts)
```

## Documentation

First `cd` into the `doc` directory and run:
```bash
make html
```

You can generate the documentation for the plugin.

First, you must have tox installed.
```bash
pip install tox
```

Then, you can run the following command with tox to generate the documentation:
```bash
tox -e docs
```
To view the generated documentation, open the following file in a browser: `BRAKET_DEFAULT_SIMULATOR_ROOT/build/documentation/html/index.html`

## Testing

Make sure to install test dependencies first:
```bash
pip install -e "amazon-braket-default-simulator-python[test]"
```

To run the unit tests:
```bash
tox
```

To run an individual test:
```bash
tox -- -k 'your_test'
```


## License

This project is licensed under the Apache-2.0 License.

