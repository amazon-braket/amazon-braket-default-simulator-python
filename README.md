**This prerelease documentation is confidential and is provided under the terms of your nondisclosure agreement with 
Amazon Web Services (AWS) or other agreement governing your receipt of AWS confidential information.**

The Amazon Braket Default Simulator is a Python open source library that provides an implementation of a quantum simulator 
that you can run locally. You can use the simulator to test quantum tasks that you construct for the [Amazon Braket SDK](https://github.com/aws/braket-python-sdk)
before you submit them to the Amazon Braket service for execution.

## Setting up Amazon Braket Default Simulator Python
You must have the [Amazon Braket SDK](https://github.com/aws/braket-python-sdk) installed before you can use the local simulator. 
Follow the instructions in the [README](https://github.com/aws/braket-python-sdk/blob/stable/latest/README.md) for setup.

**Checking the version of the DefaultSimulator**

You can check your currently installed version of `amazon-braket-default-simulator-python` with `pip show`:

```bash
pip show amazon-braket-default-simulator-python
```

or alternatively from within Python:

```
>>> from braket import default_simulator
>>> default_simulator.__version__
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
print(device.run(bell, shots=100).result().measurement_counts)
```

## Documentation

To view the documentation for the simulator, either download the .zip file or build it in your local environment.

**To download the .zip file**

Use the following command to download the .zip file
```bash
aws s3 cp s3://braket-external-assets-prod-us-west-2/sdk-docs/amazon-braket-default-simulator-documentation.zip amazon-braket-default-simulator-documentation.zip
```
Then extract the `amazon-braket-default-simulator-documentation.zip` file to your local environment. After you extract the file, open the index.html file.

**To generate the API Reference HTML in your local environment**

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
These tests will compare the performance of a series of simulator executions for your changes against the latest commit on the master branch.   
*Note*: The execution times for the performance tests are affected by the other processes running on the system.
In order to get stable results, stop other applications when running these tests.

To run an individual test:
```bash
tox -- -k 'your_test'
```


## License

This project is licensed under the Apache-2.0 License.

