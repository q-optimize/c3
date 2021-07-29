# Changelog

This Changelog tracks all past changes to this project as well as details about upcoming releases. The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the exception that the `1.x` release is still in beta with API breaking changes between minor releases. The notes below include a summary for each release, followed by details which contain one or more of the following tags:

- `added` for new features.
- `changed` for functionality and API changes.
- `deprecated` for soon-to-be removed features.
- `removed` for now removed features.
- `fixed` for any bug fixes.
- `security` in case of vulnerabilities.

- `changed` The generator now can handle any list of devices that forms a directed graph #129

## Version `1.3` - 20 Jul 2021

### Summary

Reduced coupling among different sections of the codebase by increasing the use of library style modules for all the sub-parts of the `c3-toolset`. Clean-ups also
involved renaming the Optimizers to more intuitive names. Several performance improvements were introduced by vectorized and batched operations.

### Details

- `added` A Changelog as a central location for tracking releases, new features and API breaking changes
- `added` Tests, Batch Processing for `tf_utils`, tests for `from_config()` and `asdict()` #89
- `fixed` Transmon_expanded fitting of EJ EC #89
- `added` More hjson functionality, including complex numbers #89
- `added` `fixed` Sensitivity Analysis codebase with docs and examples #125
- `changed` Names of Optimizers #120
- `added` Tests and Cleanup of `algorithms`, `qt_utils` #124 #112
- `added` Support for Python 3.9  and TF 2.4+ with more flexible dependencies #123 #95 #113 #72 #60
- `added` Tests for Calibration #119
- `added` `fixed` Model Learning codebase with docs and examples #117
- `fixed` Parsing and Reading config files - Models, tasks etc #116 #103 #98 #41
- `changed` Structure of libraries for `model`, `tf_utils`, `propagators` #99 #93
- `added` High Level Introduction to library in docs #110
- `removed` Tensorflow Optimizers that don't have correct integration #104 #124
- `added` `fixed` Hamiltonian generation and a lot more #84
- `added` Links to use binder #86 #37
- `added` `fixed` Improvements in Qiskit integration #76 #68 #59 #54 #52 #50 #48 #47 #128
- `added` Cutting the simulation space by excitation number #75
- `fixed` Fix counting the relative phase in IQ Mixing #40 
- `added` Support for Parametric gates in OpenQasm style #57
- `added` Simulation of cross talk between drive lines for Mutual Inductance #63
- `fixed` Vulnerabilities hightlighted by CodeQL #65
- `added` `fixed` Vectorization, FFT, Noise, Dressed States and a lot more #34
- `fixed` Memory Leakage caused by use of `tf.Variable` #46
- `fixed` Simulation of Tunable Coupler #45
- `added` Nightly releases as `c3-toolset-nightly` with post-release checks #42 #62
- `added` Tests for checking Notebooks #20

## Version `1.2.3` - 16 Jul 2021

### Summary

Bugfix release that addresses the memory leak due to usage of `tf.Variable` highlighted in #38 with the fix in #46.

## Version `1.2.2` - 21 Feb 2021

### Summary

Maintenance release with mostly cleanup and update of the dependencies along with some automated vulnerability checks.

### Details

- `added` Support for `tensorflow==2.4.0`, `tensorflow-probability==0.12.1`, `tensorflow-estimator==2.4.0`, `tensorboard==2.4.0`, `numpy==1.19.5`
- `fixed` Cleaned up dependencies in `requirements.txt` and `setup.py` to remove unused packages
- `fixed` Pin `qiskit` version in CI
- `added` CodeQL CI analysis for checking security vulnerabilities

## Version `1.2.1` - 2 Feb 2021

Hotfix for missing `c3.qiskit` module in the package setup.

## Version `1.2` - 29 Jan 2021

### Summary

Support for simulating many new physical elements as well as signal chains. Preliminary support for Qiskit programming with updated documentation.

### Details

- `added` Tunable Elements
- `added` Robust Optimal Control
- `added` Noise Simulation
- `added` General Signal Generation Chain
- `added` Updated Tests and Docs
- `added` Updated Examples
- `added` OpenQasm Qiskit support
- `added` SNAIL device

## Version `1.1.1` - 13 Jan 2021

### Summary

This bug-fix is a mid release before the next major release 1.2 which is expected to provide enhancements for additional devices, gates and high-level programming support. In the meantime, this bug-fix addresses the following issues.

### Details

- `fixed` Typos in version names
- `fixed` `gast` and `tensorflow` conflict due to `pip` resolver issues on Windows
- `added` Missing Templates for PRs and Issues
- `fixed` Missing `rich` in `requirements.txt`

## Version `1.1` - 22 Dec 2020 - Christmas Release

### Summary

This is the first major and properly packaged release of the `c3-toolset` library which also included a deployment to the `PyPi` package repository.

### Details

- `added` Tensorflow Optimizers - Experimental
- `added` Model Parsers
- `removed` Cleaned up Display and Plotting utils
- `added` Granular Testing
- `added` Quick Setup
- `added` Configured and Selective Testing
- `added` Faster CI/CD
- `added`, `fixed` Compatibility with Windows and MacOS
- `added` Better Examples and Docs
- `added` pip installation configs
- `added` pip-test deployment github actions