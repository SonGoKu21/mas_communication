# Public configuration

Run code commands from `code/` with `PYTHONPATH=src:.`. Experiment-specific runtimes use their own module paths as documented in the reproduction guide.

Provide credentials through environment variables or an untracked `.env` file. Never put credentials in experiment specifications or commits. The repository uses loopback endpoints, documentation addresses and generic `/opt/mas` or `/var/lib/mas` placeholders for external resources. Replace them through the relevant command-line arguments or environment configuration.

The Reddit recovery adapter requires `WEBARENA_ROOT` and the WebArena evaluator environment variables: `PLAYWRIGHT_BROWSERS_PATH`, `SHOPPING`, `SHOPPING_ADMIN`, `REDDIT`, `GITLAB`, `MAP`, `WIKIPEDIA`, and `HOMEPAGE`. Set `MAS_REDDIT_OUTPUT` to the intended output directory.

The fixtures under protection experiments are offline test inputs. Their historical admission metadata does not authorize a new execution. Live runs require a freshly validated local environment and matching source/configuration hashes.

Set `MAS_P3_OUTPUT` to the canonical output directory for the recovery-boundary experiment and pass the same path as `--output`. Resume uses the same directory and source/configuration hashes. The default is `code/results/recovery_boundary/`; experiment source directories cannot be used as outputs.
