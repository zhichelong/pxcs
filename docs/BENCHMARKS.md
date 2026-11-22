# Benchmarks

All numbers below come from one machine and one private corpus. Dataset names are replaced
by `A01`…`A15`; sizes, image counts and ratios are unchanged.

- CPU: i7-13700HX (16 cores / 24 threads)
- RAM: 32 GiB
- Storage: NVMe for `A01`–`A12`, `A14`, `A15`; one collection (`A13`) lives on a USB HDD
- Python 3.11.9, PXCS 1.4.0
- Harness: `tools/pxcs_suite.py`, archives written to a temporary directory and deleted
  after each run; every archive passed `verify`.

## Corpus

`Before` = ratio with the pre-optimization strategy (per-image codecs, no clustering,
no JPEG XL). `—` means the collection was added to the harness after that baseline.

| ID | Content | Size | Images | Before | After | Create |
|---|---|---:|---:|---:|---:|---:|
| A01 | JPEG, no variants | 0.50 GiB | 148 | 98.92% | **83.49%** | 18.2 s |
| A02 | PNG, no variants | 0.56 GiB | 75 | 84.20% | 84.20% | 36.0 s |
| A03 | PNG, few variants | 0.84 GiB | 611 | 85.80% | **73.40%** | 37.5 s |
| A04 | JPEG, no variants | 0.85 GiB | 164 | 95.28% | **79.19%** | 49.1 s |
| A05 | JPEG, few variants | 1.62 GiB | 157 | 70.85% | **57.28%** | 64.7 s |
| A06 | JPEG, no variants | 1.95 GiB | 450 | 98.38% | **81.50%** | 50.6 s |
| A07 | PNG, few variants | 2.31 GiB | 315 | 84.33% | **63.30%** | 219.9 s |
| A08 | JPEG, medium | 3.47 GiB | 565 | 50.20% | **39.01%** | 262.2 s |
| A09 | PNG, many variants | 3.68 GiB | 397 | 39.65% | **33.68%** | 132.0 s |
| A10 | JPEG, variant-heavy | 3.92 GiB | 620 | 96.73% | **22.60%** | 147.5 s |
| A11 | JPEG, uneven | 4.25 GiB | 281 | 63.97% | **52.75%** | 191.0 s |
| A12 | PNG, many variants | 5.36 GiB | 1 164 | 52.38% | **40.28%** | 217.0 s |
| A13 | PNG, extra large (HDD) | 10.59 GiB | 723 | 79.33% | **52.27%** | 1 271.9 s |
| A14 | JPEG, huge | 14.33 GiB | 3 078 | — | **30.39%** | 686.8 s |
| A15 | JPEG, huge | 29.93 GiB | 5 842 | 56.17% | **51.74%** | 1 750.0 s |

Total: **84.16 GiB, 14 590 images, 5 134 s** of `create` (≈ 16.8 MB/s average), all
archives verified with 0 failures.

## Where the bytes go

Aggregate for `A14` + `A15` (43.3 GiB of source, 16.7 GiB archived):

| Reason | Entries | Source | Archived | Share of archive |
|---|---:|---:|---:|---:|
| `residual_lossy` | 6 671 | 34 380 MiB | 12 239 MiB | 60.3% |
| `key_lossy_passthrough` | 1 796 | 8 608 MiB | 6 840 MiB | 33.6% |
| `residual_larger_than_source` | 226 | 1 096 MiB | 1 096 MiB | 4.3% |
| `zlib` (non-image files) | 71 | 343 MiB | 342 MiB | 1.5% |

`key_lossy_passthrough` means "this resolution appears exactly once in the collection", so
there is nothing to delta against; those entries are compressed with the `jpegxl` codec
(79.5% of source). `residual_larger_than_source` is the fallback when a delta turns out
not to be smaller than the original.

For collections without any variants (`A01`, `A04`, `A06`), JPEG XL alone accounts for the
whole improvement: 98.9% → 83.5%, 95.3% → 79.2%, 98.4% → 81.5%.

## Throughput and resource use

| Metric | Value |
|---|---|
| `create` throughput | 16.8 MB/s average over the corpus |
| CPU utilization | 27–45% of 24 logical threads |
| Peak working set | 5.0–8.1 GiB (budget = 60% of physical RAM) |
| `verify`, 397-entry archive | 147.3 s → 59.8 s with 8 threads (2.46×) |
| `extract`, same archive | 621.2 s → 107.8 s with 8 threads (5.76×) |
| JPEG XL encode | ≈ 9 MB/s per thread |
| JPEG XL decode | ≈ 11 MB/s per thread |

Peak memory tracks the budget rather than the dataset size: the memory guard throttles
in-flight work instead of letting the process be killed by the OS.

## Known ceiling

`create` does not saturate the machine. The CPU-time profile of a JPEG collection is about
83% libjxl (a C library) and 17% Python glue; wall-clock efficiency is limited further by
the ordered commit path, which waits for the oldest in-flight group before submitting
more. See [VALIDATION.zh-CN.md](VALIDATION.zh-CN.md) for the measurements. Closing that gap
is tracked work, not a language limitation of the archive format itself.

## Reproducing

The corpus is private. To run the same harness on your own collections:

```powershell
copy datasets.example.json datasets.json   # then edit paths
python -m tools.pxcs_suite --list
python -m tools.pxcs_suite --all --out report.json
```

`datasets.json` is git-ignored.

## 1.4.1: reference tree vs star topology

1.4.1 replaces the per-group star (every variant deltas against one key) with a
reference tree: each member points at the most similar member planned before it,
depth capped at 6 for lossy residuals and 8 for lossless ones.

Nine collections, 2026-11-20, same machine and thread count (12). `1.4.0` is the
released GitHub build; `1.4.1` is this change.

| ID | Content | Size | 1.4.0 | 1.4.1 | Delta |
|---|---|---:|---:|---:|---:|
| A16 | PNG, full variant collection | 53.9 GiB | — | **16.3%** | — |
| A17 | PNG, one character, 177 variants | 1.3 GiB | 26.8% | **10.2%** | -16.6 pt |
| A18 | JPEG, medium variants | 10.6 GiB | 66.7% | **59.3%** | -7.3 pt |
| A19 | PNG, mixed | 2.1 GiB | 42.0% | **37.2%** | -4.8 pt |
| A20 | JPEG, medium variants | 2.7 GiB | 16.1% | **11.8%** | -4.3 pt |
| A21 | JPEG, large variants | 13.8 GiB | 31.1% | **28.6%** | -2.4 pt |
| A22 | PNG, mixed | 3.7 GiB | 39.5% | **38.1%** | -1.4 pt |
| A23 | PNG, loose images | 1.7 GiB | 77.0% | 76.2% | -0.7 pt |
| A24 | JPEG, loose images | 3.1 GiB | 80.8% | 80.6% | -0.2 pt |
| A25 | JPEG, loose images | 1.5 GiB | 79.3% | 79.3% | 0.0 pt |

**Loose images gain nothing** — there is no neighbour to delta against, and the
extra thumbnail pass costs a little. Variant-heavy collections gain 1.4–16.6
points, and the gain scales with how many variants share the same base drawing.

Speed is not monotonic. Smaller residuals compress faster, so most variant sets
got *faster* (A18 and A22 gained the most); a few lost ground
because the tree encodes one depth layer at a time.

The cost is decode depth: a cold single-image extraction of a depth-6 entry takes
about 2.2x as long as a key frame. Sequential extraction reuses the parent cache.


## 1.5.0: hardened reading, memory and speed adjustments

The format and compression ratios are unchanged; what this release verifies is that
hardening did not slow the normal path.

| Metric | 1.4.1 | 1.5.0 |
|---|---:|---:|
| Archive size (60 JPEGs / 758 MiB) | 50.48 MiB | 50.48 MiB (+1 B from the random archive_id) |
| Create time (interleaved A/B, two runs) | 37.9 / 37.1 s | 37.1 / 37.0 s |
| `verify` full | 64.4 s | 58.0 s |
| `verify` quick | 62.4 s | 0.35 s |
| Create peak RSS | 5.9 GiB | 5.35 GiB |
| Reference-tree planning peak (n=1000 / n=3000) | ~2 GB / 32 GB temporary | 5.9 MB / 17.7 MB |

The `verify --quick` gap is not a 180x speedup: 1.4.1's quick still reconstructed residual
chains (contradicting its help text), while 1.5.0's quick only checks payload CRC/SHA.
That is a semantics fix, not a pure optimization. On the same archive, full verification
still decodes everything and is about 10% faster.


## 1.6.0: cross-group key references

One 91-image / 1.75 GiB artwork folder, `--key-ref-scope off` (default) vs explicit `folder`:

| Metric | off | folder |
|---|---:|---:|
| Keys | 7 | 4 |
| Key bytes | 55.04 MiB | 32.73 MiB |
| Archive size | 138.62 MiB | 131.19 MiB (**-5.4%**) |
| Create time | 177.3 s | 202.3 s (about +14%) |

The three cross-mode RGBA keys became deltas of RGB keys; their keyframes totalled
22.31 MiB and the deltas 14.88 MiB, so the actual saving is 7.43 MiB. Full verify was
91/91 and extracted pixels/alpha matched.
