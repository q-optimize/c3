# Contributing to C3 Development

- [Contributing to C3 Development](#contributing-to-c3-development)
  - [Where to Start](#where-to-start)
  - [Opening a New Issue](#opening-a-new-issue)
  - [Forking and Cloning the Repository](#forking-and-cloning-the-repository)
  - [Remotes and Branches](#remotes-and-branches)
  - [Setting up the development environment](#setting-up-the-development-environment)
    - [Using conda](#using-conda)
    - [Using virtualenv](#using-virtualenv)
    - [Development on Apple Silicon](#development-on-apple-silicon)
      - [Developer Setup](#developer-setup)
      - [User Setup](#user-setup)
  - [Coding Standards](#coding-standards)
  - [Tests and Test Coverage](#tests-and-test-coverage)
    - [Running Tests Locally](#running-tests-locally)
  - [Type Annotations](#type-annotations)
  - [Doc Strings & Documentation](#doc-strings--documentation)
    - [Building and Viewing Documentation Locally](#building-and-viewing-documentation-locally)
  - [Pull Requests](#pull-requests)
    - [Changelog](#changelog)
  - [Continuous Integration Checks](#continuous-integration-checks)
  - [Contributor License Agreement](#contributor-license-agreement)
  - [Git Flow Development Style](#git-flow-development-style)
  - [Developer Tools and Tips](#developer-tools-and-tips)

This guide assumes you have a working Python distribution (native or conda) and some basic understanding of Git and Github.

Check the instructions for [installing](https://docs.conda.io/en/latest/miniconda.html#linux-installers) and [using](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) a Conda or a [Native](https://realpython.com/installing-python/) Python distribution. If you are on Windows, we recommend a **non** Microsoft Store installation of a full native Python distribution. The [Atlassian Git tutorials](https://www.atlassian.com/git/tutorials) is a good resource for beginners along with the online [Pro Git Book](https://git-scm.com/book/en/v2).

## Where to Start

As a first-time contributor to the C3 project, the best place to explore for possible contributions is the [Issues](https://github.com/q-optimize/c3/issues) section. Please go through the existing open and closed issues before opening a new one. Issues that would allow a newcomer to contribute without facing too much friction are labelled [`good-first-issue`](https://github.com/q-optimize/c3/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). The documentation and examples are other places that could always use some extra help. Check the [`documentation`](https://github.com/q-optimize/c3/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation) label for suggestions on what needs work. We would also be very happy to review PRs that add tests for existing code, which happens to be a great way to understand and explore a new codebase.

If you are more adventurous and willing to dive deeper into the codebase, you will find several possible contribution opportunities annotated with a `TODO` comment and a short explanation of what needs to be done. Refer to the [API documentation](https://c3-toolset.readthedocs.io/en/latest/c3.html) for the relevant section and open an issue if you would like to tackle any of these `TODO`s.

## Opening a New Issue

Any new feature or bug-fix that is added to the codebase must typically be first discussed in an Issue. If you run into a bug while working with C3 or would like to suggest some new enhancement, please [file an issue](https://github.com/q-optimize/c3/issues/new/choose) for the same. Use a helpful title for the Issue and make sure your Issue description conforms to the template that is automatically provided when you open a new Bug report or Feature request. One of the core developers will either address the bug or discuss possible implementations for the enhancement.

## Forking and Cloning the Repository

In order to start contributing to C3, you must fork and make your own copy of the repository. Click the fork icon in the top right of the repository page. Your fork is a copy of the main repo, on which you can make edits, which you can then propose as changes by making a pull request. Once forked, you should find a repo like `https://github.com/githubusername/c3`. The next step involves making a local copy of this forked repository using the following command:

```bash
git clone https://github.com/githubusername/c3
```

## Remotes and Branches

Once you have cloned your repository, you want to set up [branches](https://www.atlassian.com/git/tutorials/using-branches) and [remotes](https://www.atlassian.com/git/tutorials/syncing).

Remotes let you backup and sync your code with an online server. When you clone the repository, git automatically adds a `remote` called `origin` which is your fork of the C3 repo. You typically also want to add a remote of the parent C3 repository. The convention for naming the parent repository is `upstream` and you set this up with the commands below:

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

Branches let you work on a new feature or bug-fix without affecting the `master` version of your code. You should **always** create a branch when you start working on something new. Branches in C3 have the following stucture:

- `master` - The current stable release, source code of the `pip` package release
- `dev` - Development branch for ongoing work, might contain bugs or breaking changes
- `docs` - Everything related to adding docs, which is also rendered on [RTD](https://c3-toolset.rtfd.io)
- `feature` - Personal branches to work on new features or bug fixes

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

We describe below the steps for setting up your development environment for locally building and testing C3, using either `conda` or `virtualenv`.

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

This will create an editable installation of `c3-toolset` in the `c3-dev` or `env` environment which is linked to your local `c3/` directory making sure that any changes you make to the code are automatically included. For development purposes, you will probably want to additionally install `jupyter`, `notebook`,  `matplotlib`, `pytest`, `pytest-xdist` and `pytest-cov`. If you wish to use Qiskit with C3, you must also install `qiskit`. The `requirements.txt` file lists out these dependencies with their versions.

### Development on Apple Silicon

If you are planning to use or develop `c3-toolset` on Apple Silicon devices (eg, the M1 Macs), you need to follow a slightly more involved setup process. This is because several python packages do not distribute pre-built binaries for Apple Silicon using standard `pip`/`conda` repositories and must be obtained separately. This is true eg, for `tensorflow` which is a hard dependency or for `qiskit` which is required if you want to run circuits. 

#### Developer Setup

1. Install and setup Miniforge on your Apple Silicon Mac from [here](https://github.com/conda-forge/miniforge)
2. Create a new conda environment (`conda create --name=c3-dev python=3.8`)
3. Install the latest tensorflow from conda-forge `conda install tensorflow -c conda-forge`
4. Use `pip` to install the corresponding version of `tensorflow-probability` that matches the version of `tensorflow` from the previous step
4. Install separately the different qiskit submodules (instead of the metapackage). Instructions [here](https://github.com/Qiskit/qiskit/issues/1201) 
4. Use `pip` to install the rest of the requirements from the `requirements.txt` file after relaxing all the pinned versions 
5. Edit the `setup.py` file to remove all the package dependencies by removing the `install_requires` argument 
6. Install `c3-toolset` in development mode using `pip install -e .` from the root of the repository

#### User Setup
1. Follow steps 1 to 4 from the Developer Setup above
2. Use `pip install c3-toolset` to install the package
3. If you need to run `qiskit` specific functionalities, follow step 5 above

## Coding Standards

We use `pre-commit` hooks to enforce coding standards. C3 uses `black` for code formatting, `mypy` for type checking and `flake8` as a linter for static checking. Pre-commit hooks provide a seamless way to integrate these three tools in your development workflow. Follow the steps below to set it up:

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

Untested code is bad code. There is a constant effort to increase the test coverage for the C3 codebase. In this light, any new code being pushed must come with accompanying tests. If you are writing a new feature, make sure you add detailed tests that check that your code does what is expected of it by checking the output for known typical and edge cases. If you are fixing a bug, write regression tests to ensure the bug doesn't pop up again in a later release due to changes elsewhere in the code. Additionally, we actively welcome contributions that aren't a bug-fix/feature but include tests to increase the coverage of the current codebase.

Our test-suite is built using `pytest` and is stored inside `test/` in the root of the repository. Using `pytest`, writing tests is a straightforward task as outlined below:

- Import the C3 modules that you wish to test inside your new file
- Write functions of the form `test_some_feature()`
- Create a new file `test/test_some_c3_module.py`
- These functions can either check some individual feature (unit test) or the interaction between a variety of features (integration test)
- Testing is done using `assert` statements which check some output against a known value

For more inspiration, check some of the existing tests. We encourage you to actively use pytest fixtures, parameters, and markers, to make testing efficient, readable and robust. Details on these features is beyond the scope of this document, so we ask you to refer [here](https://realpython.com/pytest-python-testing/).

### Running Tests Locally

As discussed later, we have automated testing on remote servers set up for our codebase on Github. Here we outline how to run the tests and check test coverage locally.

At any moment, you can run the whole suit of tests from the root of the C3 repository using the command below:

```bash
pytest -v --cov=c3 test/
```

The `-v` flag enables verbosity providing useful insights on why some tests failed. The `--cov=c3` flag will check the coverage of tests for our `c3/` codebase and generate a detailed report showing the coverage for individual files. By default, outputs to console from the code are disabled when running tests. You can use the `-s` flag to enable these outputs.

Typically when developing, you do not want to run all the tests everytime (because it can be quite slow at times). This is where filtering tests is useful.

- **Name-based filtering**: You can limit pytest to running only those tests whose fully qualified names match a particular expression. You can do this with the `-k` parameter.
- **Directory scoping**: By default, pytest will run only those tests that are in or under the current directory or any directory explicitly provided by you
- **Test categorization**: pytest can include or exclude tests from particular categories that you define. You can do this with the `-m` parameter. Markers are listed in [`pytest.ini`](pytest.ini)
- **Explicit filenames**: You can run only a single test file by explicitly mentioning the file name instead of a directory

## Type Annotations

Python is a dynamically typed language which is great when you are solving a homework problem and your codebase is a single `.py` monolith but it breaks pretty soon when you start working in large distributed teams. Hence we use type annotations to make everyone's life easier when they are either contributing to the C3 codebase or building code on top of it. Type annotations are also very easily integrated with IDEs (discussed later) and guide type checkers (such as `mypy`) to preemptively detect possible errors.

Example type annotation:

```python
def func(x: tf.constant) -> tf.constant:
  """
  Do cool TensorFlow Stuff
  """
  # fancy tensorflow GPU autodiff tensor operations
  return tf.constant(result)
```

Check [this resource](https://realpython.com/python-type-checking/#annotations) for more details on Type Checking and Annotation.

## Doc Strings & Documentation

If untested code is bad code, undocumented code is unusable code. Besides striving for generous inline comments, mnenomic variable names and readable implementations, we extensively document C3 through docstrings. These docstrings follow the [numpydoc docstring](https://numpydoc.readthedocs.io/en/latest/format.html) format and are automatically picked up by our documentation building tool (Sphinx) to generate the API documentation. A sample docstring is shown below:

```python
def func(arg1: float) -> int:
  """Short Note

  Parameters
  -----------
  arg1: float
      1st argument to the function
  
  Returns
  -----------
  int
      Processed result
  
  Raises
  -----------
  ZeroDivisionError
      Throws error when dividing by zero
  """
```

Additionally, you can include a detailed note after `Raises` with an example code snippet showing the usage of specific module. Besides these module docstrings, we also include example documentation which is found inside the `docs/` directory. These are reStructured Text `.rst` files which are also parsed and rendered by Sphinx. You can find more details in the [relevant documentation](https://www.sphinx-doc.org/en/master/). The [Sphinx cheatsheet](https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/CheatSheet.html) is handy when writing example documentations. Please contact one of the core developers or open an issue if you are planning to add example documentation and would need some help.

### Building and Viewing Documentation Locally

In order to build and view the docs locally, you need to perform the following steps:

```bash
pip install sphinx sphinx-rtd-theme autoapi autodoc
sudo apt install graphviz

cd docs/
make clean
make html
```

This will render the documentation into html files (it can take a few minutes) which can then be viewed by launching a simple python server:

```bash
cd docs/_build/html
python -m http.server 4242
```

You can now open `localhost:4242` in your browser and view the docs as they would be rendered online. For checking a live version of the online docs as rendered from the `master`, `dev` or `docs` branch, head over to [c3-toolset.rtfd.io](https://c3-toolset.rtfd.io).

## Pull Requests

Once you feel your work is ready to be merged into the main codebase, open a Pull Request to merge your feature branch to the `dev` branch on `q-optimize/c3`. Please write a useful title and a detailed description for your Pull Request.

Follow the *What, Why, How, Remarks* structure when writing the description of your Pull Request. In the remarks section, add comments to help the reviewer or explain possible quirks/issues with your implementation or aspects that might require extra attention. For new features, please add code snippets that demonstrate how this feature is to be used. Wherever possible, add screenshots and GIFs so that it's convenient to quickly review your PR.

Make sure you [allow maintainers to edit your branch](https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/) when you are creating a Pull Request from a fork. Then, anyone with Write access to the upstream repository will be able to add commits to your branch. This can make the review process easier for maintainers; sometimes it’s quicker for a maintainer to make a small change themselves instead of asking you to make the change.

You can keep your Pull Request as a Draft as long as you feel you still need to make changes. It is often advisable to open a PR when you are maybe 85% done with the work, which gives maintainers enough time to check what's going on and guide you if something needs correction. When you are ready, request for review on your PR. Your PR must be approved by at least 2 core developers before it can be merged. The Code review might involve suggested changes, which you can incorporate by committing and pushing more code in the same feature. You **DO NOT** need to open a new PR or create a new branch if someone requests for changes on your Pull Request. If your PR is not reviewed by any of the core developers within 3 days, please request politely in the comments with a follow-up.

### Changelog

We use the [CHANGELOG.md](CHANGELOG.md) as a living document that not only records past changes in various releases but also tracks development of bug fixes, new features, API changes etc for the upcoming release. Every Pull Request should update the Changelog with a short note explaining the addition and noting the PR number. When making a new release, this gets updated with the version number and date; and is also included in the description for Github Releases. The next time a PR is opened for merging into `dev`, a new Upcoming Release section should be added to the top of the Changelog.

## Continuous Integration Checks

As previously discussed, tests are an inherent part of good code. All Pull Requests made to the repository are automatically tested on Windows, MacOS and Ubuntu with Python 3.7, 3.8 and 3.9, which come up in the PR as `Checks`, notifying you whenever a particular check is failing on any platform. Except the `Code Complexity`, all checks should be passing for your PR to be ready for review/merge. The `Codecov` bot will also comment on the PR with a report on the change in the test coverage for that particular PR. If your PR reduces the overall test coverage of the codebase, it is not yet ready and you need to add more tests.

## Code Reviews

Code Reviews are an integral aspect of our development workflow with the explicit goal of ensuring that every new Pull Request either maintains or preferably improves the overall code quality of this codebase. We try to automate large chunks of the traditional code review process through the PR checks (formatters, linters, tests, code-coverage, code-climate etc) while expecting human reviewers to focus more on things lower in the [Code Review Pyramid](https://www.morling.dev/blog/the-code-review-pyramid/). This generally means that code reviewers shouldn't be nitpicking on style or tests but address possible issues in the design and implementation of the bugfix/feature. Everyone is welcome and encouraged to participate in our code-review process. Particularly, new contributors have a lot to gain by reviewing Pull Requests since it gives them a crash course in what it means to add a new feature to this codebase and what are the general pitfalls one needs to look out for. A good introduction to code reviews is the [Google Engineering Practices Documentation](https://google.github.io/eng-practices/). This [blog](https://stackoverflow.blog/2019/09/30/how-to-make-good-code-reviews-better/) on StackOverflow has further insights on improving the process.

## Contributor License Agreement

Before you can submit any code, all contributors must sign a contributor license agreement (CLA). By signing a CLA, you’re attesting that you are the author of the contribution, and that you’re freely contributing it under the terms of the Apache-2.0 license. Additionally if you are contributing as part of your employment, you will need to state that you have obtained necessary permissions from your employer to contribute freely to the C3 project.

When you contribute to the C3 project with a new pull request, a bot will evaluate whether you have signed the CLA. If required (typically for first time contributors), the bot will comment on the pull request, including a link to electronically accept and sign the agreement. The individual CLA document will be available for review as a PDF in the comment.

## Git Flow Development Style

We follow the [`git-flow`](https://nvie.com/posts/a-successful-git-branching-model/) style development in C3. Along with [`semver`](https://semver.org/), this translates to the following branching and release structure for the repository:

- We maintain a semantic versioning based release numbering.
- Stable/Production ready code goes into `master`.
- We build features by making branches off the `dev`, working on a feature and merging it back into `dev`.
- When we are ready to make a release (maybe once every month), i.e, **release code** into `master` and `pip`, we create a new branch `release/x.y.z`  off `dev`, where we usually have a list of To-Do for pre-release tasks (called a release sprint/final run). We make all changes into this and then **release** by merging this `release/x.y.z`  into `master` (with no `fast-forward`, so making a merge commit) and then merging `master` back into `dev` (`fast-forward` with no merge commit). This cycle continues for every release.
- When we find a bug in the stable release and need to make a `hotfix`, we branch off `master`, as `hotfix/x.y.z+1`; make changes and merge (`--no-ff`) it into `master` and then merge (`--ff`) `master` into `dev`.

This would ensure we have a clean release cycle, hotfixes are quick and well maintained and the master is always stable.

Semantic versioning in short would mean the following -
For version `x.y.z`, we don't change `x` unless the API is going to break in a backwards *incompatible* way. We change `y` for major releases with lots of new features and change `z` for bug hotfixes and minor releases.

## Developer Tools and Tips

The general philosophy of development for C3 is captured in the following points:

- DRY (Do not Repeat Yourself) - Inherit and extend, Import and reuse.
- TDD (Test Driven Development) - Add tests for everything.
- Do one thing well - Write atomic code that only solves 1 problem at a time.
- Open-Closed Principle - Open to extension, but Closed to modification.
- Useful Docstrings - Include code snippets and technical summary.
- Correct Physics - Ensure technical consistency and add ref paper & notes.

It is useful to configure and use an IDE for your development purposes, since that greatly augments your experience and will often seamlessly integrate the various tools mentioned previously. We list the relevant tools for some of the common IDEs below:

- VSCode - [Docstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring), [PyRight](https://github.com/microsoft/pyright), [Formatting](https://code.visualstudio.com/docs/python/editing#_formatting), [Linting](https://code.visualstudio.com/docs/python/linting), [Testing](https://code.visualstudio.com/docs/python/testing)
- PyCharm - [Docstring](https://www.jetbrains.com/help/pycharm/creating-documentation-comments.html), [Type Hinting](https://www.jetbrains.com/help/pycharm/type-hinting-in-product.html#typeshed), [Formatting](https://www.jetbrains.com/pycharm/guide/tips/reformat-code/), [Testing](https://www.jetbrains.com/help/pycharm/pytest.html)
- Atom - [Docstring](https://github.com/spadarian/docblock-python), [Type Hinting](https://github.com/MagicStack/MagicPython), [Type Checking](https://github.com/elarivie/linter-mypy), [Testing](https://github.com/pghilardi/atom-python-test)
- Sublime Text - [Docstring](https://packagecontrol.io/packages/DocBlockr_Python), [Formatting, Type Hints &amp; Checking](https://packagecontrol.io/packages/Anaconda)
