# Contributing

Bug reports and focused pull requests are welcome. Before opening a change:

1. Create an isolated Python 3.9+ environment.
2. Install development dependencies with `python -m pip install -e '.[test]'`.
3. Add or update synthetic tests for behavioral changes.
4. Run `pytest -q`.

Do not submit benchmark structures, activity labels, embedding caches, model
checkpoints, whitening artifacts, credentials, or machine-specific paths.

Changes to the published protocol, search space, or frozen parameters must be
explicitly identified and must not overwrite the corresponding publication
artifact without provenance.

By submitting a contribution, you agree that it may be distributed under the
repository's `LicenseRef-TACTIVS-Commercial-Notice-1.0` terms. Maintainers may
request an additional contributor agreement before accepting substantial
changes that affect future licensing rights.
