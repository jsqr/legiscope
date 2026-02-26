# Sample Data

This directory contains small raw samples from several municipalities,
git-tracked so new contributors can run the pipeline without configuring
a DVC remote. All sample locality names are prefixed with "Test" so they
are easy to filter out of real analysis.

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

## Usage

1. Copy a sample into the data directory:

```bash
cp -r sample_data/IL data/laws/
```

2. Initialize the jurisdiction (if not already registered).
   The default `params.yaml` already targets IL/TestChicago/municipal-code:

```bash
python scripts/init.py
```

3. Run the pipeline:

```bash
dvc repro
```
