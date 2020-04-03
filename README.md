**This prerelease documentation is confidential and is provided under the terms of your nondisclosure agreement with 
Amazon Web Services (AWS) or other agreement governing your receipt of AWS confidential information.**

The Amazon Braket Default Simulator is a Python open source library that provides an implementation of a quantum simulator 
that you can run locally. You can use the simulator to test quantum tasks that you construct for the [Amazon Braket SDK](https://github.com/aws/braket-python-sdk)
before you submit them to the Amazon Braket service for execution.

## Prerequisites

### Python 3.7.2 or greater
Download and install Python 3.7.2 or greater from [Python.org](https://www.python.org/downloads/).
If you are using Windows, choose **Add Python to environment variables** before you begin the installation.

### Amazon Braket SDK
Download and install the Amazon Braket SDK. Follow the instructions in the [README](https://github.com/aws/braket-python-sdk/blob/stable/latest/README.md).	

## Setting up Amazon Braket Default Simulator Python
Clone or download the amazon-braket-default-simulator-python repo to your local environment. 

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
The quantum simulator implementation `DefaultSimulator` plugs into the `LocalSimulator` interface in 
[Amazon Braket SDK](https://github.com/aws/braket-python-sdk) using the `backend` value as `"default"`. 

**Executing a circuit using the DefaultSimulator**
```python
from braket.circuits import Circuit
from braket.devices import LocalSimulator

device = LocalSimulator("default")

bell = Circuit().h(0).cnot(0, 1)
print(device.run(bell, shots=100).result.measurement_counts)
```

## Documentation

To generate the docs locally, you must have tox installed.
```bash
pip install tox
```
First `cd` into the `doc` directory and run:
 ```bash
 make html
 ```
Then, you can run the following command with tox to generate the documentation:
```bash
tox -e docs
```
This generates the documentation in a `/build` subfolder. To view the generated documentation, 
open the following file in a browser: `../build/documentation/html/index.html`

## Testing

If you want to contribute to the project, be sure to run unit tests and get a successful result 
before you submit a pull request. To run the unit tests, first install the test dependencies using the following command:
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

