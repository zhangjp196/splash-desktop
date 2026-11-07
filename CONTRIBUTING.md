# Contributing

Issues are welcome here: bugs, model requests, and questions about running
Splash on your machine. Include the Splash version, the Mac and macOS version,
and the model you were serving.

Pull requests are reviewed and merged in this repository. Open them against
`main`; for a larger change, open an issue first so we can agree on the
approach before you build it. Describe in the pull request what changed and
how you tested it: `make check` needs no model weights, but it runs the Metal
tests, so it needs a Mac that Splash supports; the real-model checks a change
may need are in [Validate](docs/validate.md).

## External branches

Do not merge an external branch into this repository. That includes upstream,
forks, and topic branches from another tree: `git merge`, `git pull`, and any
merge commit after the `1.7.1` release are not accepted, even when the result
would look correct. Port the change by hand instead: read the other tree,
apply the parts this one needs in its own modules, keep the local tests and
documentation consistent, and make a normal linear commit. `make check-source`
runs `dev/tools/check_history.py`, which fails on a merge after `1.7.1`.

[The documentation](docs/README.md) covers building from source, configuration,
the API, the models Splash loads, its internals and the test suite.
