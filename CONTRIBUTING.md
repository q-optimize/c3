# Contributing to $C^3$ Development

This guide assumes you have a working Python distribution (native or conda) and some basic understanding of Git and Github. 

Check the instructions for [installing](https://docs.conda.io/en/latest/miniconda.html#linux-installers) and [using](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) a Conda or for a [Native](https://realpython.com/installing-python/) Python distribution. If you are on Windows, we recommend a **non** Microsoft Store installation of a full native Python distribution. The [Atlassian Git tutorials](https://www.atlassian.com/git/tutorials) is a good resource for beginners along with the online [`Pro Git Book`](https://git-scm.com/book/en/v2).

## Where to Start

As a first-time contributor to the $C^3$ project, the best place to explore for possible contributions is the [Issues](https://github.com/q-optimize/c3/issues) section. Please go through the existing open and closed issues before opening a new one. Issues that would allow a newcomer to contribute without facing too much friction are labelled [`good-first-issue`](https://github.com/q-optimize/c3/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). The documentation and examples are other places that could always use some extra help. Check the [`documentation`](https://github.com/q-optimize/c3/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation) label for suggestions on what needs work. We would also be very happy to review PRs that add tests for existing code, which happens to be a great way to understand and explore a new codebase.

## Opening a New Issue

Any new feature or bug-fix that is added to the codebase must typically be first discussed in an Issue. If you run into a bug while working with $C^3$ or would like to suggest some new enhancement, please [file an issue](https://github.com/q-optimize/c3/issues/new/choose) for the same. Use a helpful title for the Issue and make sure your Issue description conforms to the template that is automatically provided when you open a new Bug report or Feature request. One of the core developers will either address the bug or discuss possible implementations for the enhancement.

## Forking and Cloning the Repository

In order to start contributing to $C^3$, you must fork and make your own copy of the repository. Click the fork icon in the top right of the repository page. Your fork is a copy of the main repo, on which you can make edits, which you can then propose as changes by making a pull request. Once forked, you should find a repo like `https://github.com/githubusername/c3`. The next step involves making a local copy of this forked repository using the following command:

```bash
git clone https://github.com/githubusername/c3
```

## Remotes and Branches

Once you have cloned your repository, you want to set up [branches](https://www.atlassian.com/git/tutorials/using-branches) and [remotes](https://www.atlassian.com/git/tutorials/syncing).

Remotes let you backup and sync your code with an online server. When you clone the repository, git automatically adds a `remote` called `origin` which is your fork of the $C^3$ repo. You typically also want to add a remote of the parent $C^3$ repository. The convention for naming the parent repository is `upstream` and you set this up with the commands below:

```bash
# navigate inside your local clone
cd c3/
git remote add upstream https://github.com/q-optimize/c3
git remote -v
```

This will show you the two remotes currently setup:

```bash
origin  https://github.com/githubusername/c3.git (fetch)
origin  https://github.com/githubusername/c3.git (push)
upstream        https://github.com/q-optimize/c3 (fetch)
upstream        https://github.com/q-optimize/c3 (push)
```

Branches let you work on a new feature or bug-fix without affecting the `master` version of your code. You should **always** create a branch when you start working on something new. Branches in $C^3$ have the following stucture:

* `master` - The current stable release, source code of the `pip` package release
* `dev` - Development branch for ongoing work, might contain bugs or breaking changes
* `docs` - Everything related to adding docs, which is also rendered on [RTD](https://c3-toolset.rtfd.io)
* `feature` - Personal branches to work on new features or bug fixes

You should start your development by making a new `feature` branch off `dev`, as described below:

```bash
git checkout dev
git pull upstream dev
git checkout -b feature dev
git push --set-upstream origin feature
```

The golden rule is to always work on an atomic piece of code wihtin a feature branch. So you do not want to work on completely different sections of the codebase while building a new feature. However, it is possible that others might also be working on same sections of the code in parallel. When their code gets merged into `dev`, you typically want to pull in these changes into your `feature` branch with the following commands:

```bash
git checkout dev
git pull upstream dev
git checkout feature
git merge dev
git push origin feature
```

Make frequest commits, write useful and sensible commit messages and keep your local code synced with your fork and changes pulled in from the parent repository.

## Setting up the development environment

We describe below the steps for setting up your development environment for locally building and testing $C^3$, using either `conda` or `virtualenv`.

### Using conda

```bash
conda create --name=c3-dev python=3.8
cd c3/ # navigate to the root of your local repository
conda activate c3-dev
pip install -e .
```

### Using virtualenv

```bash
cd c3/ # navigate to the root of your local repository
python3 -m venv env
source env/bin/activate
pip install -e .
```

This will create an editable installation of `c3-toolset` in the `c3-dev` or `env` environment which is linked to your local `c3/` directory making sure that any changes you make to the code are automatically included. For development purposes, you will probably want to additionally install `jupyter`, `notebook`,  `matplotlib`, `pytest`, and `pytest-cov`. If you wish to use Qiskit with $C^3$, you must also install `qiskit==0.23`.

## Coding Standards

We use `pre-commit` hooks to enforce coding standards. $C^3$ uses `black` for code formatting, `mypy` for type checking and `flake8` as a linter for static checking. Pre-commit hooks provide a seamless way to integrate these three tools in your development workflow. Follow the steps below to set it up:

```bash
conda activate c3-dev # or source env/bin/activate
pip install pre-commit
pre-commit install
```

This will install the pre-commit hooks for your local git repository. When you make a commit the next time, `pre-commit` will spend a few minutes to download and install `black`, `flake8` and `mypy` environments for testing your commits meet the requirements enforced using these tools. This initial setup is only done once and subsequent commits won't run this installation process. `black` will reformat your code and the other tools will throw errors where there is a possible type violation or syntax error. You must remember to stage any changes made at this point before trying to commit your code again. So it typically goes in the following order:

```bash
# you write some cool code
git add c3/cool-c3-feature.py
git commit -m "made cool new hack"
.
.
.
# black says it reformatted the code
# flake8 shows unused imports
# mypy shows type violations
# you fix issues highlighted above
git add c3/cool-c3-feature.py
git commit -m "made cool new hack"
# all pre-commit hooks pass
# commit successful!
```

## Tests and Test Coverage

Untested code is bad code. There is a constant effort to increase the test coverage for the $C^3$ codebase. In this light, any new code being pushed must come with accompanying tests. If you are writing a new feature, make sure you add detailed tests that check that your code does what is expected of it by checking the output for known typical and edge cases. If you are fixing a bug, write regression tests to ensure the bug doesn't pop up again in a later release due to changes elsewhere in the code. Additionally, we actively welcome contributions that aren't a bug-fix/feature but include tests to increase the coverage of the current codebase.

Our test-suite is built using `pytest` and is stored inside `test/` in the root of the repository. Using `pytest`, writing tests is a straightforward task as outlined below:

* Create a new file `test/test_some_c3_module.py`
* Import the $C^3$ modules that you wish to test inside your new file
* Write functions of the form `test_some_feature()`
* These functions can either check some individual feature (unit test) or the interaction between a variety of features (integration test)
* Testing is done using `assert` statements which check some output against a known value

For more inspiration, check some of the existing tests. We encourage you to actively use pytest fixtures, parameters, and markers, to make testing efficient, readable and robust. Details on these features is beyond the scope of this document, so we ask you to refer [here](https://realpython.com/pytest-python-testing/).

### Running Tests Locally

As discussed later, we have automated testing on remote servers set up for our codebase on Github. Here we outline how to run the tests and check test coverage locally.

At any moment, you can run the whole suit of tests from the root of the $C^3$ repository using the command below:

```bash
pytest -v --cov=c3 test/
```

The `-v` flag enables verbosity providing useful insights on why some tests failed. The `--cov=c3` flag will check the coverage of tests for our `c3/` codebase and generate a detailed report showing the coverage for individual files. By default, outputs to console from the code are disabled when running tests. You can use the `-s` flag to enable these outputs.

Typically when developing, you do not want to run all the tests everytime (because it can be quite slow at times). This is where filtering tests is useful. 

* **Name-based filtering**: You can limit pytest to running only those tests whose fully qualified names match a particular expression. You can do this with the `-k` parameter.
* **Directory scoping**: By default, pytest will run only those tests that are in or under the current directory or any directory explicitly provided by you
* **Test categorization**: pytest can include or exclude tests from particular categories that you define. You can do this with the `-m` parameter. Markers are listed in [`pytest.ini`](pytest.ini)
* **Explicit filenames**: You can run only a single test file by explicitly mentioning the file name instead of a directory

## Type Annotations

Python is a dynamically typed language which is great when you are solving a homework problem and your codebase is a single `.py` monolith but it breaks pretty soon when you start working in large distributed teams. Hence we use type annotations to make everyone's life easier when they are either contributing to the $C^3$ codebase or building code on top of it. Type annotations are also very easily integrated with IDEs (discussed later) and guide type checkers (such as `mypy`) to preemptively detect possible errors.

Example type annotation:

```python
def func(x: tf.Variable) -> tf.Variable:
  """
  Do cool TensorFlow Stuff
  """
  # fancy tensorflow GPU autodiff tensor operations
  return tf.Variable(result)
```

Check [this resource](https://realpython.com/python-type-checking/#annotations) for more details on Type Checking and Annotation.

## Doc Strings & Documentation

### Building and Viewing Documentation Locally

## Pull Requests

## Continuous Integration Checks

## Contributor Licence Agreement

## Git Flow Development Style

## Developer Tools and Tips

- [x] Finding Issues to Contribute
- [x] Opening and Discussing new Issues
- [x] Forking and Cloning the Repo
- [x] Setting up the development environment
- [x] Setting up and using Pre-Commit Hooks
- [x] Branches and remotes
- [x] Tests and Test Coverage
- [x] Running Tests Locally
- [x] Type Annotations
- [ ] Doc Strings
- [ ] Building and Viewing Documentation locally
- [ ] Pull Request Best Practices
- [ ] CI Checks
- [ ] CLA Signing
- [ ] Git flow development style for Releases
- [ ] Developer FAQ - common gotchas, IDE development & extension philosophy