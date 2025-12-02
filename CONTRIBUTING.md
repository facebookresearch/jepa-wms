# Contributing to JEPA-Internal
We want to make contributing to this project as easy and transparent as
possible. Codebase norms are described at https://fburl.com/gdoc/kk9rd2ul.

## Pull Requests
We welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code is consistent with style guidance (below) and lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").
7. Add reviewer(s) for approval.

## Pre-Commit

pre-commit hooks are useful to find linter/formatting issues before submitting code for review:

```bash
pip install pre-commit  # make sure pre-commit is available in your env.
pre-commit install  # install pre-commit hooks - you only need to do this once.
```

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* 4 spaces for indentation rather than tabs
* 119 character line length
* PEP8 formatting

See https://fburl.com/gdoc/kk9rd2ul for detailed guidance, required for changes to `src/`
and other code that is not project-specific.

## License
By contributing to this repository, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
