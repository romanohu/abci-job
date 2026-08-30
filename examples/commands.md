# Command examples

The command begins after `--`. Its tokens are preserved and quoted separately
in the generated script.

```bash
python submit.py \
  --config configs/abci_default.toml \
  --name package-command \
  --dry-run \
  -- \
  python -m package.train --output "results/run one"
```

```bash
python submit.py \
  --config configs/abci_default.toml \
  --name shell-command \
  -- \
  bash scripts/run.sh --label "two words"
```

Arguments that look like submitter options remain part of the command when
they appear after `--`:

```bash
python submit.py \
  --config configs/abci_default.toml \
  --name option-command \
  --dry-run \
  -- \
  python -m package.train --dry-run --label "run one"
```

## Full-node experiment manifest

Use a separate manifest for one to eight independent single-GPU commands. The
ABCI configuration used here must set `queue = "rt_HF"`.

```bash
python submit_many.py \
  --config configs/abci_default.toml \
  --experiments experiments/example.toml \
  --name example-batch \
  --dry-run \
  --print-script
```
