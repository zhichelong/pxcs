# Contributing

Thanks for taking a look. This is a small, opinionated project; the rules are short.

## Priorities

**Correctness > compression ratio > speed.** A change that makes archives smaller but
weakens a guarantee (byte-exact restore, deterministic payloads, abort-instead-of-truncate)
will be rejected.

## Before opening a PR

1. Run the full check from the repository root:

   ```powershell
   python -m pip install -r requirements-dev.txt
   python -m tools.check
   ```

   It runs the unit tests and the golden vectors. `--coverage` also prints line coverage.

2. Add a test for every behaviour change. Bug fixes need a test that fails before the fix.
3. Do not change the golden vectors. If a codec change is unavoidable, it needs a format
   minor bump and a migration note in [CHANGELOG.md](CHANGELOG.md).
4. Keep archives readable: a new reader must open every archive written by an older writer.

## Style

- Comments, docstrings and commit messages may be **Chinese or English**; mixed is fine.
  User-facing strings must go through `language/<code>.json`, never inline.
- Match the surrounding code. There is no formatter gate beyond 4-space indent and 88-96
  column lines.
- Windows-only code paths should be marked; if you add a POSIX path, guard it and note it
  in the README platform section.

## Documentation

Docs are bilingual: `X.md` is English, `X.zh-CN.md` is Chinese. When you change a spec,
update both, or open the PR with a note that the translation is pending.

## What is hard to accept

- New runtime dependencies. The tool ships as a single Windows executable; every
  dependency is a size and licensing cost.
- Changes to the file layout without a version bump.
- Heuristics that trade a guarantee for a better ratio.
