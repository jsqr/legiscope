# Sample Data

This directory contains small raw samples from several municipalities,
git-tracked so new contributors can run the pipeline without configuring
a DVC remote. All sample locality names are prefixed with "Test" so they
are easy to filter out of real analysis.

All samples contain `.docx` source files in their `raw/` directories.

## Sample jurisdictions

| State | Locality | Code slug |
|-------|----------|-----------|
| AK | TestKingCove | code-of-ordinances |
| AR | TestCaveSprings | code-of-ordinances |
| AZ | TestApacheJunction | city-code |
| CA | TestAdelanto | municipal-code |
| CO | TestAkron | town-code |
| CT | TestEastLyme | code-of-ordinances |
| IL | TestChicago | municipal-code |

## Batch parse all samples

Run the parse stage for every sample jurisdiction at once:

```bash
./scripts/parse_samples.sh
```

Preview what would be done without executing anything:

```bash
./scripts/parse_samples.sh --dry-run
```

The script will, for each sample:
1. Copy the sample into `data/laws/`
2. Convert the `.docx` to `code.txt` via `convert_docx.sh`
3. Initialize the jurisdiction via `init.py`
4. Run the DVC `parse` stage

## Manual usage (single sample)

1. Copy a sample into the data directory:

```bash
cp -r sample_data/IL data/laws/
```

2. Convert DOCX to TXT:

```bash
bash scripts/convert_docx.sh data/laws/IL/TestChicago/municipal-code/raw
```

3. Initialize the jurisdiction (if not already registered).
   The default `params.yaml` already targets IL/TestChicago/municipal-code:

```bash
python scripts/init.py
```

4. Run the pipeline:

```bash
dvc repro
```
