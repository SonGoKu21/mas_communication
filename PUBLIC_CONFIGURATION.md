# Configuring the public artifact

The repository contains generic deployment examples, not a configuration for an existing server.

- `MAS_CODE_ROOT`: repository location. Shell entry points derive this from their own location unless overridden.
- `MAS_DATA_ROOT`: absolute directory for datasets, models, results, and caches. Set this explicitly before storage setup or model-download scripts.
- `MAS_PYTHON`: Python executable for local shell runners (default: `python3` from the active environment).
- `MAS_VLLM`: vLLM executable (default: `vllm`). `MAS_VLLM_ACTIVATE` optionally selects an activation script; `MAS_LIBSTDCXX` optionally selects a compatibility library.
- `MAS_CREDENTIALS_FILE`: private environment file required by the legacy DeepSeek runtime template. Keep this outside the repository.

For example, choose your own storage directory, then run:

```bash
export MAS_DATA_ROOT="/absolute/path/to/experiment-data"
bash configure-storage.sh
source server-env.sh
```

Storage setup refuses relative data paths and refuses to replace existing directories or unrelated symlinks.

Remaining `/opt/mas` and `/var/lib/mas` paths are generic installation templates. Adapt deployment units and pass the Python entry points' path arguments for your environment. `192.0.2.10` is a documentation-only address: replace benchmark URLs with your own endpoints before live execution. Loopback addresses denote explicitly local services.

Run `python3 scripts/audit_public_paths.py` before publishing. This scans tracked text, including legacy copies, for personal home directories, numbered storage mounts, and non-example IP addresses. It is a targeted configuration audit, not a guarantee that arbitrary files contain no identifying information.

Changes to the current tree do not remove older Git commits. Anonymous review snapshots should be generated from the current cleaned tree and pinned to the intended commit.
