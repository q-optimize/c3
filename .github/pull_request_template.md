## What
Describe very concisely what this Pull Request does. 

## Why
Describe what motivated this Pull Request and why this was necessary. Use closing keywords to link to the relevant Issue. Ex. Closes #666

## How
Describe details of how you implemented the solution, outlining the major steps involved in adding this new feature or fixing this bug. Provide code-snippets if possible
showing example usage.

## Remarks
Add notes on possible known quirks/drawbacks of this solution. If this introduces an API-breaking change, please provide an explanation on why it is necessary to break API
compatibility and how users should update their notebook/script workflows once this PR is merged.

## Checklist
Please inculde and complete the following checklist. Your Pull Request is (in most cases) not ready for review until the following have been completed. You can create a draft 
PR while you are still completing the checklist. Check the [Contribution Guidelines](https://github.com/q-optimize/c3/blob/dev/CONTRIBUTING.md) for more details. 
You can mark an item as complete with the `- [x]` prefix

- [ ] Tests - Added unit tests for new code, regression tests for bugs and updated the integration tests if required
- [ ] Formatting & Linting - `black` and `flake8` have been used to ensure styling guidelines are met
- [ ] Type Annotations - All new code has been type annotated in the function signatures using type hints
- [ ] Docstrings - Docstrings have been provided for functions in the `numpydoc` style
- [ ] Documentation - The tutorial style documentation has been updated to explain changes & new features
- [ ] Notebooks - Example notebooks have been updated to incorporate changes and new features
