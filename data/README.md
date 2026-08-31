# Recall datasets

The `recall/` directory contains the ten CSV files found in the source research
directory. Files named `<length>.csv` use different relations; files named
`<length>_same_relation.csv` use a repeated relation. Every file has these columns:

- `input`: prompt containing context facts and a partial query;
- `target_position`: one-based position of the queried fact;
- `completion`: expected continuation.

Each target position has 50 examples, giving 400, 800, 1,600, 3,200, and 6,400 rows
for lengths 8, 16, 32, 64, and 128, respectively, per variant.

Important: the exploratory source did not include a generator, license, or provenance
record for these CSVs. They are copied byte-for-byte for reproducibility. Add the
dataset-generation procedure and appropriate attribution before publishing an
archival artifact.

Verify that the packaged files are unchanged:

```bash
(cd data && sha256sum -c SHA256SUMS)
```
