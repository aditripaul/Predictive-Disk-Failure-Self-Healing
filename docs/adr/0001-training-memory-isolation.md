# ADR 0001: Process isolation and scratch placement in train_model

Status: accepted (history recorded 2026-10-07; moved out of `pipelines/train_model.py`
docstring, which is unchanged in behaviour).

## Context

The training pipeline ran out of memory on fleet-scale data. The stage split
(assemble, extract-split per split, orchestrate) is the result. The cap that
motivated it was RLIMIT_AS, which counts reserved address space, so it was
superseded by an RSS-based cap (`src/resource_limits.py`).

## Decision and history

History, since the reasoning generalizes beyond this specific pipeline:
attempt 1 put the join alone in its own subprocess, leaving `main()` to
collect all three splits itself - failed on real data because collecting
all three as full eager DataFrames, then a second numpy copy of each
while the DataFrame was still alive, needed several times the size of
even the largest split. Attempt 2 narrowed and sequenced that collection
(one split at a time, minimal columns, immediately freed) but still ran
it in `main()` - failed anyway, because each split's collect inherited
whatever high-water mark the earlier splits' collects had left behind in
that same process, even though any one split alone fit comfortably.
Attempt 3 moved ALL split collection into the join's own subprocess -
failed again, because the join's OWN retained memory was enough to starve
the very first split collected right after it. Attempt 4 isolated EVERY
distinct heavy Polars collect into its own process - the join, and each
split separately - and got further, but a plain `scan_parquet(...)
.filter(...).collect()` still failed even in a fully isolated process
(attempt 5's fix: row-group-native reads, `_row_group_parts`), and even
THAT still failed once fixed to spill each row group to disk instead of
accumulating results in memory (attempt 6). Peak-RSS instrumentation
(`resource.getrusage` checkpoints, since removed) finally pinned
attempt 6's failure exactly: forming ONE
combined Polars DataFrame from all the spilled parts, then converting
THAT to a numpy array, needed both the ~6GB combined DataFrame and a
further ~3.7GB array alive at once - together enough to fail even though
either one alone fit easily. Attempt 7 (`_build_feature_arrays`) builds
the numpy array directly from the spilled parts instead, one part at a
time, never forming a combined DataFrame at all.

The lesson generalizes beyond "isolate the heavy step" (attempts 1-4) to
"the SHAPE of how a result is built matters, not just which process it
runs in" (attempts 5-7): reading a wide file needs explicit row-group
discipline, a loop must spill and drop each iteration's result rather
than accumulate them, and converting a large result to a different
representation (Parquet -> Polars -> numpy) can itself double memory if
the intermediate representation lingers - build directly into the final
form instead of combining-then-converting through one.

Attempt 7 fixed the memory problem (train, then validation, completed
successfully for the first time), but immediately surfaced an unrelated
one: `np.save`'s underlying `array.tofile()` failed with a PARTIAL write
- the OS ran out of disk space partway through, independent of this
pipeline's own memory cap. `work_dir` was living under `tempfile`'s
default location (`/tmp`), which turned out too small to hold
`frame.parquet` (tens of GB at fleet scale) plus every split's own
artifacts at once. Fixed by putting `work_dir` under `resource_limits.
scratch_dir` (`_scratch_base`) instead - `<gold_dir>/../tmp` by default,
a filesystem already proven to have room for comparably large files -
and by spilling each split's own row-group parts into a subdirectory of
`work_dir` rather than a separate `tempfile.TemporaryDirectory()` (which
would have defaulted right back to `/tmp`).

## Consequences

- The stage split still lowers real peak memory, but it is no longer required
  for the cap itself. Removing it is a candidate simplification, to be done only
  after a full local run reproduces the baseline output bit-for-bit.
- `work_dir` lives under `resource_limits.scratch_dir`, not `/tmp`, because
  `/tmp` was too small for the artifacts.
