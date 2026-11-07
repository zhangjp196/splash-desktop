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

## Engineering rules

- Clean and general: one rule for every case, no special cases, flags or
  hacks; a change deletes the mechanism it supersedes.
- A fixed rule with measured feedback first; an estimator only after an A/B
  shows it beats a simple baseline.
- Tests assert behaviour, not implementation.
- Comments give the rule, its reason and one number, and cite nothing outside
  the repository.
- Five layers, each depending only on those below it. The frontend (`server/`,
  `splash`, `install/`, `macos/`: HTTP APIs, chat templates, tool calls, the
  installer and the app) reaches the engine through the wire protocol alone.
  The engine (`runtime/engine/`: requests, scheduling, memory, cache) uses the
  model through its interface headers alone, except engine assembly
  (`Bootstrap`, `NativeArguments`, `RuntimeResources`), the one part that
  builds the concrete model. The model (`runtime/model/`: architectures,
  formats, weights and the runtime that encodes commands), ops
  (`runtime/ops/`: operators and their kernel selection), and kernels
  (`runtime/metal/`, `*.metal`: Metal shaders and their ABI) follow.
- `make architecture-check` enforces these; its exception lists only shrink.
