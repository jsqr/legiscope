# Sample Data

This directory contains a small raw sample from the WindyCity dataset,
git-tracked so new contributors can run the pipeline without configuring
a DVC remote.

## Usage

1. Copy the sample into the data directory:

```bash
cp -r sample_data/IL data/laws/
```

2. Initialize the jurisdiction (if not already registered).
   The default `params.yaml` already targets IL/WindyCity/municipal-code:

```bash
python scripts/init.py
```

3. Run the pipeline:

```bash
dvc repro
```
