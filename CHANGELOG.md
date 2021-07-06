# Changelog

This Changelog tracks all past changes to this project as well as details about upcoming releases. The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the exception that the `1.x` release is still in beta with API breaking changes between minor releases. The notes below are in chronologically reverse order and for each version, you will find a summary, followed by details which contain one or more of the following tags:

- `added` for new features.
- `changed` for functionality and API changes.
- `deprecated` for soon-to-be removed features.
- `removed` for now removed features.
- `fixed` for any bug fixes.
- `security` in case of vulnerabilities.

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