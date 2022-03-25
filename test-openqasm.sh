coverage run -m pytest test/unit_tests/braket/default_simulator/openqasm/ \
test/unit_tests/braket/default_simulator/test_openqasm_state_vector_simulator.py
coverage combine
coverage report --include="src/braket/default_simulator/openqasm/*",src/braket/default_simulator/openqasm_state_vector_simulator.py