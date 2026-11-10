# Package

Release archives contain no Hugging Face credentials.
Users accessing private models supply their own `HF_TOKEN` or Hugging Face login.

Release versions are three-part, `x.y.z`, with no `v` prefix: after `1.2.1`
comes `1.2.2` for a fix or `1.3.0` for a feature. Use the same version in all
three commands:

```sh
VERSION=1.2.2  # the release being built
make package RELEASE_VERSION=$VERSION
make package-bottle RELEASE_VERSION=$VERSION
make package-check RELEASE_VERSION=$VERSION
```

`make package` writes the runtime archive,
`dist/splash-<version>-arm64-macos26.tar.gz`, its checksum beside it
(`<archive>.sha256`) and the formula, `dist/splash.rb`; `make package-bottle`
adds the bottle to `dist/` and its block to the formula. These commands do not
publish. Build bottles on the oldest supported macOS. Bottle/check commands use
a temporary tap and remove their installation; they refuse to replace an
existing Splash installation. The install check requires a poured bottle and
runs the bundled launcher without a compiler or separate Python installation.

The runtime package allowlists engine, Python, server and launcher files; tests,
benchmarks and developer documents are excluded. User model links survive
upgrades; downloads remain in the Hugging Face cache.

Before publishing, verify that the models the README names and the draft
repositories of `families.FAMILIES` are publicly accessible. Check the installed
bottle on a supported Mac without developer tools; building from the source tree
is not an installation check.

## Publish

A release is made from its annotated tag, after `make package` and, for the
desktop app, `make package-dmg` ([macOS app and disk image](#macos-app-and-disk-image))
have left the command line's files of its version in `dist/`:

```sh
git tag -a $VERSION -m $VERSION    # on the release commit
make package RELEASE_VERSION=$VERSION
make package-dmg RELEASE_VERSION=$VERSION
```

Create a GitHub Release tagged `<version>` with the runtime archive, its
`.sha256` file, the disk image and its `.sha256` file. The disk image is ad hoc
signed and not notarized: a downloaded copy opens with Open Anyway in System
Settings › Privacy & Security. `dev/tools/package_dmg.py` writes the image but
not its checksum, so write that beside it with `shasum -a 256` (a bare file
name, so `shasum -c` works from the download directory).

Open a pull request in `incoai/homebrew-tap` replacing `Formula/splash.rb` with
`dist/splash.rb`; its URLs must point to the published release assets. After the
tap update merges, verify a fresh install and an upgrade from the previous
release through the public tap, including a real model request and preservation
of user data. Run `brew audit --strict --online incoai/tap/splash`.

`make release-check` and the other real-model targets check a build before it
is published ([Validate](validate.md#release-check)). Keep release notes and
local paths outside the source tree.

## Desktop app

`macos/` is the Splash desktop app, a SwiftUI control panel for the Mac
distributed as a disk image:

```sh
make package-app RELEASE_VERSION=$VERSION   # dist/Splash Desktop.app
make package-dmg RELEASE_VERSION=$VERSION   # and Splash-Desktop-<version>.dmg
```

`dev/tools/package_dmg.py` builds the app with `swift build`, puts the runtime
`make package` ships under `Contents/Resources/runtime`, and writes the disk
image. `RUNTIME_DIR=<directory>` overrides the archive with a staged runtime
directory, for a source checkout without one, and `SOURCE_OVERLAY=1` embeds
this checkout's `install/` and `server/` instead of the archive's, so a
published engine serves the working tree's installer and server code.

The app is called Splash Desktop: the bundle is `Splash Desktop.app`, the
executable inside it is `SplashDesktop`, and the volume is
`Splash Desktop <version>`. Its bundle identifier is still `ai.inco.splash`,
so an update installs beside an older `Splash.app` rather than over it, and
everything the runtime addresses by name — the `splash` command, the engine,
the archive, the tap, the data directory and the settings database — is
unchanged; delete the old app once the new one is in place.

Neither the app nor the disk image is code-signed or notarized: it is for local
installation, and distributing it requires signing and notarization. The app is
built with Xcode's Swift toolchain, and the placeholder icon is generated at
package time. `make package-app` builds just `dist/Splash Desktop.app` (no
DMG).

## Private test releases

Testers can install a build before it is public, from a private Hugging Face
repository:

```sh
make package RELEASE_VERSION=$VERSION
make publish-test RELEASE_VERSION=$VERSION TEST_RELEASE_REPO=owner/splash-releases
```

`make publish-test` (`dev/tools/publish_test.py`) uploads the archive, its
checksum, the tester installer `dev/tools/install.sh` and `latest`, which
names this version, to `TEST_RELEASE_REPO` with the developer's own
`hf auth login`, creating the repository as a private one if it does not
exist; `publish_test.py --no-latest` uploads without moving `latest`. It
prints the command testers run with a read token for the repository in
`SPLASH_TOKEN`: curl reads the token's header from its standard input, so the
token is in no command line or URL. The installer requires an Apple Silicon
Mac on macOS 26.4 or newer. It downloads the version `latest` names, or
`SPLASH_VERSION`, checks its SHA-256, unpacks it under
`~/Library/Application Support/Splash/app`, points `current` at it, writes a
`splash` command into `/opt/homebrew/bin` or `/usr/local/bin` when writable,
else `~/.local/bin` (`SPLASH_BIN_DIR` chooses), and removes older versions. It
refuses to replace a `splash` command it did not write, and to switch versions
while a server runs. Running it again upgrades; installed models stay. To try
an archive before uploading it, serve `dist/` (`python3 -m http.server`) and
run the installer with `SPLASH_BASE_URL` naming that server, `SPLASH_VERSION`
and any `SPLASH_TOKEN`.

## macOS app and disk image

`make package-dmg` packages a self-contained `Splash-Desktop-<version>.dmg`: the
native SwiftUI control panel (`macos/`) with the runtime archive embedded
under `Contents/Resources/runtime`, so installing the app needs no other
download and neither does first run (only the chosen model does):

```sh
make package RELEASE_VERSION=1.0.0
make package-dmg RELEASE_VERSION=1.0.0     # needs the archive above
```

The app owns no model or engine code: it locates the embedded runtime,
launches `install/launcher.py serve`, shows its log and a live panel of
`/status` metrics (decode/prefill rates, request counts, Metal memory, caches
and admission waits), and keeps a menu bar item to start and stop it. The
window opens on the live panel, with control and the log behind it in the tab
band under the header; the header itself carries only the state and the
actions. Its thermal page holds the [heat-and-noise control](api.md#heat-and-noise) of
the Mac itself rather than of a model, in three modes — the system default,
which passes nothing; `fixed`, whose balanced, moderate or quiet levels are
the ladder's rungs and pass `--thermal-policy` to a server it starts; and
`smart`, the app's own pacer, computing a duty from the fans and temperatures
it reads each second, with a high, medium or low level that sets how hard the
fans may work before prefill gives way. The choice is remembered in
`UserDefaults` (as the interface language is) and reaches a running engine at
once through `POST /v1/thermal`. Beside it
the page shows the fans and the CPU, GPU, IO and disk temperatures the engine
reads. Its settings live in SQLite at
`~/Library/Application Support/ai.inco.splash/settings.db` (one key/value row
per setting, `SPLASH_SETTINGS_DB` points it elsewhere): the panel saves each
edit after a short pause, the next launch opens on what the last one saved, and
Restore Defaults returns the form to the values the first launch opened with
(upstream model, port 8000, 8-bit KV, a 28 GiB memory ceiling, a 204800 token
context, an 8-request admission limit, a 10 minute idle release (with a switch
to keep the weights resident), a switch for responsive prefill (off), and the
SSD cache off — turning it on gives it a 16 GiB quota and a 32 GiB a day write
budget). A
stored value the launcher would reject falls
back to its first-launch default rather than starting a server that refuses to
run. `swift run --selfcheck-settings` exercises the store and that round trip
headlessly against a temporary database. The **Models** pane lists the saved
configurations and shows the one that is running above them: add, edit and
switch between them, or aim one at a model folder on your disk. A
configuration keeps what belongs to its model — the target, its files and
text-only mode, the DFlash2 draft and speculation, the memory ceiling, context
length and streamed stores — and how that model's engine runs: KV and
recurrent-state precision, the neural engine, admission, prefill pacing and
thinking defaults. The **Settings** tab holds what every configuration shares:
**Connection** (port, LAN access and extra Host names), **Access** (API key,
served names and request limits), and **Storage** (idle release, the SSD
cache, its quota, write budget and eviction). Its **Application** section
holds the window's appearance and interface language, and the folder and Hub
every download uses. The server page saves as it is edited and applies at the
next start, so a new configuration does not repeat what the app already knows;
application choices apply at once. The
conversation is the server's own web page, embedded in the **Chat** tab: the
app loads `chat.html?embed=1`, where the page keeps only the conversation and
posts its history and state to the view, so the native band can list chats,
start a new one, and clear them with one undo. The same page without `embed`
still opens in a browser, sidebar and all, for anyone using the server
directly. Its labels and
the chat page follow the system interface language (English and Simplified
Chinese are bundled). The app offers
`install/suggested-models.tsv`, per model the variant meant for the Mac's
memory with its download size and a few words about the model, and starts
coding agents with `splash AGENT --port PORT` in Terminal, in a folder the user
picks, from a script that holds the running server's API key and removes itself
as it starts. Its launch options are every option `splash serve` takes but the
ones of one model (`--revision`, `--draft-model`, `--mmproj`) and the ones it
sets itself, from one table (`LaunchOption.all`) the tests hold to `splash
serve --help`; the API key reaches the server as `SPLASH_API_KEY`, from the
Keychain, and nothing starts while the Keychain does not give it. It serves
model files on this Mac the user adds with `splash serve --model local/NAME
--model-dir PATH`, NAME being the file's or folder's name made a repository
name, and at launch it runs `install/cached.py`. While a model runs, the Server
shows its context limit, from `/v1/models`, and its `/status`: requests,
answering and prompt speed, time to first token, the tokens generated, read and
taken from the cache, and the Metal memory it holds, with each request's line
of the server's log, folded.
