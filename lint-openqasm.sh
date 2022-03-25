isort src/braket/default_simulator/openqasm \
src/braket/default_simulator/openqasm_state_vector_simulator.py \
test/unit_tests/braket/default_simulator/openqasm \
test/unit_tests/braket/default_simulator/test_openqasm_state_vector_simulator.py \
demo

black src/braket/default_simulator/openqasm \
src/braket/default_simulator/openqasm_state_vector_simulator.py \
test/unit_tests/braket/default_simulator/openqasm \
test/unit_tests/braket/default_simulator/test_openqasm_state_vector_simulator.py \
demo

flake8 src/braket/default_simulator/openqasm \
src/braket/default_simulator/openqasm_state_vector_simulator.py \
test/unit_tests/braket/default_simulator/openqasm \
test/unit_tests/braket/default_simulator/test_openqasm_state_vector_simulator.py \
demo
