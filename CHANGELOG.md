# Changelog

## v1.15.0 (2023-06-12)

### Features

 * add optional third angle to MS gate

## v1.14.0.post0 (2023-05-25)

### Documentation Changes

 * add a linter to check proper rst formatting and fix up incorrect docs

## v1.14.0 (2023-05-15)

### Features

 * update local sim properties to include supported modifiers

## v1.13.3 (2023-05-10)

### Bug Fixes and Other Changes

 * New implementation for helper `_get_rabi_dict` in rydberg AHS.

## v1.13.2 (2023-05-01)

### Bug Fixes and Other Changes

 * Modification to `scipy_integrate_ode_run` In braket_ahs local simulator

## v1.13.1 (2023-04-26)

### Bug Fixes and Other Changes

 * test: parallelize test execution for pytest

## v1.13.0 (2023-04-20)

### Features

 * optimize performance for simulating control modifiers

## v1.12.3 (2023-03-29)

### Bug Fixes and Other Changes

 * Revert threshold to switch AHS solvers

### Testing and Release Infrastructure

 * add dependabot updates for GH actions

## v1.12.2 (2023-03-27)

### Bug Fixes and Other Changes

 * update default step value in doc string for the rydberg simulator

## v1.12.1 (2023-03-14)

### Bug Fixes and Other Changes

 * ahs local simulator update

## v1.12.0 (2023-03-03)

### Deprecations and Removals

 * deprecate python 3.7

### Bug Fixes and Other Changes

 * Use `singledispatchmethod` from functools

### Documentation Changes

 * Remove Black badge

## v1.11.5.post0 (2023-02-13)

### Testing and Release Infrastructure

 * update github workflows for node12 retirement

## v1.11.5 (2023-02-09)

### Bug Fixes and Other Changes

 * update: adding build for python 3.10

## v1.11.4 (2023-01-16)

### Bug Fixes and Other Changes

 * tweak noise on sv test

## v1.11.3 (2023-01-05)

### Bug Fixes and Other Changes

 * numpy ragged array error

## v1.11.2 (2023-01-04)

### Bug Fixes and Other Changes

 * remove oq3 named sims

## v1.11.1 (2022-12-13)

### Bug Fixes and Other Changes

 * remove warning for binary expressions

## v1.11.0 (2022-12-07)

### Features

 * Adjoint Gradient changes

### Bug Fixes and Other Changes

 * Relax pydantic version constraint

## v1.10.2 (2022-11-22)

### Bug Fixes and Other Changes

 * regex escape char in unit test

## v1.10.1.post0 (2022-11-21)

### Testing and Release Infrastructure

 * Remove Ocean plugin from dependent tests

## v1.10.1 (2022-11-15)

### Bug Fixes and Other Changes

 * Reference code from the current commit for dependent tests

## v1.10.0 (2022-10-31)

### Features

 * neutral atom simulator

## v1.9.1 (2022-10-27)

### Bug Fixes and Other Changes

 * Allow verbatim pragma

## v1.9.0 (2022-09-15)

### Features

 * update antlr version to 4.9.2

## v1.8.1 (2022-08-31)

### Bug Fixes and Other Changes

 * add native gates to device properties

## v1.8.0 (2022-08-29)

### Features

 * add ionq native gates
 * optimize interpretation by enabling direct computation of bu…

## v1.7.2.post0 (2022-08-10)

### Testing and Release Infrastructure

 * Don't run tests on push to feature branches
 * Add SF plugin to dependent tests

## v1.7.2 (2022-08-05)

### Bug Fixes and Other Changes

 * correct behavior for result types all

## v1.7.1 (2022-08-04)

### Bug Fixes and Other Changes

 * rebuild parsers with new antlr version

## v1.7.0 (2022-08-04)

### Features

 * Simulation of OpenQASM programs

## v1.6.3 (2022-08-04)

### Bug Fixes and Other Changes

 * remove pytester
 * bump pytest version further
 * modify pytester reference to _pytest.pytester
 * pin pytest version for pytester plugin
 * Enable simulation of OpenQASM programs 🚀 (#💯)

## v1.6.2 (2022-04-19)

### Bug Fixes and Other Changes

 * align ECR gate definition with OQC

## v1.6.1 (2022-04-15)

### Bug Fixes and Other Changes

 * ensure correct behavior for target ordering with DensityMatrix r…

## v1.6.0 (2022-04-12)

### Features

 * add ECR gate (#5)

## v1.5.0 (2022-02-01)

### Features

 * adding two qubit Pauli channels

## v1.4.0 (2022-01-27)

### Features

 * added controlled-sqrt-not gate.

### Bug Fixes and Other Changes

 * Add controlled-sqrt-not gate to dm simulator

## v1.3.0 (2021-08-23)

### Features

 * Calculate arbitrary observables

## v1.2.3 (2021-08-12)

### Bug Fixes and Other Changes

 * Calculate exact statistics from trace

### Documentation Changes

 * Doc corrections

## v1.2.2 (2021-08-06)

### Bug Fixes and Other Changes

 * Reconcile supported operations with managed format

## v1.2.1 (2021-05-26)

### Bug Fixes and Other Changes

 * Fix perf tests

## v1.2.0 (2021-05-24)

### Features

 * Density matrix simulator

### Testing and Release Infrastructure

 * Use GitHub source for tox tests

## v1.1.1.post2 (2021-03-19)

### Testing and Release Infrastructure

 * Run unit tests for dependent packages

## v1.1.1.post1 (2021-03-11)

### Testing and Release Infrastructure

 * Add Python 3.9

## v1.1.1.post0 (2021-03-03)

### Testing and Release Infrastructure

 * Add team to CODEOWNERS
 * Use main instead of PyPi for build dependencies

## v1.1.1 (2021-01-27)

### Bug Fixes and Other Changes

 * Make tensor products more efficient

## v1.1.0.post2 (2021-01-12)

### Testing and Release Infrastructure

 * Enable Codecov

## v1.1.0.post1 (2020-12-30)

### Testing and Release Infrastructure

 * Add build badge
 * Use GitHub Actions for CI

## v1.1.0.post0 (2020-12-04)

### Testing and Release Infrastructure

 * Change tox basepython to python3

## v1.1.0 (2020-11-26)

### Features

 * Always accept identity observable factors

## v1.0.1.post0 (2020-10-30)

### Testing and Release Infrastructure

 * updating CODEOWNERS

## v1.0.1 (2020-10-29)

### Bug Fixes and Other Changes

 * Enable simultaneous measurement of observables with shared factors

## v1.0.0.post2 (2020-09-09)

### Documentation Changes

 * Add README badges
 * Get Read the Docs working
 * add readthedocs link

### Testing and Release Infrastructure

 * Add CHANGELOG.md
 * Update formatting to follow new black rules
 * Automatically publish to PyPi
 * add stale workflow and labels to issues templates

## v1.0.0.post1 (2020-08-14)

The only way to update a description in PyPi is to upload new files;
however, uploading an existing version is prohibited. The recommended
way to deal with this is with
[post-releases](https://www.python.org/dev/peps/pep-0440/#post-releases).

## v1.0.0 (2020-08-13)

This is the public release of the Amazon Braket Default Simulator!

The Amazon Braket Default Simulator is a Python open source library that provides an implementation of a quantum simulator that you can run locally.
