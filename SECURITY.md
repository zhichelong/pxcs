# Security policy

## Supported versions

Only the current minor (1.6) is supported. Format minors 0–4 can be *read*; only the
newest writer is maintained.

## Reporting a vulnerability

Open a private security advisory on GitHub (Security → Report a vulnerability). Please do
not open a public issue for a security problem.

Include: the archive or a minimal reproducer, the exact command, the Python version, and
the version shown by `pxcs.py --version`. Encrypted-archive issues are easiest to
reproduce with a small archive you are willing to share.

## Design

- **KDF**: Argon2id (default 256 MiB, 3 iterations, 4 lanes), stored parameters, 16-byte
  salt. `scrypt` is available as a fallback when the `cryptography` build lacks Argon2id.
- **AEAD**: AES-256-GCM, a fresh 96-bit nonce per message, no nonce reuse across payloads.
- **AAD binding**: every encrypted message is bound to the archive `archive_id`, so
  ciphertext cannot be transplanted between archives or entries without detection.
- **Domain separation**: index, payloads and the password check use distinct AAD tags.
- **Integrity**: payloads additionally carry CRC32 + SHA-256 of the plaintext, verified by
  `verify` and on extraction.

## Threat model

Protected against:

- an attacker who obtains the archive file but not the passphrase;
- tampering: modified, truncated, reordered or transplanted ciphertext is rejected;
- offline guessing beyond what the KDF cost allows (Argon2id at the default parameters).

**Not** protected against:

- **Metadata.** The archive's total size and volume count are visible. Unless filename
  encryption is enabled (`--encrypt-names`), the index is plaintext and reveals relative
  paths, file sizes and image dimensions.
- **Traffic analysis.** Entry count and payload size distribution leak information about
  the collection.
- **Passphrase loss.** There is no recovery path and no key escrow. Losing the passphrase
  means losing the archive.
- **A compromised machine.** Malware, keyloggers and memory inspection are out of scope.

## Operational notes

- Passphrases are read from a prompt, a file (`--password-file`) or stdin
  (`--password-stdin`). Avoid passing them as command-line arguments; ARG_MAX is visible
  to other users on some systems.
- Archives are verified with `pxcs.py verify` after writing; an interrupted create writes
  nothing usable rather than a partial archive.
- Password files should live outside any directory you archive.


## Reading untrusted archives

Archives may come from untrusted sources. Three hard limits apply before any decoding or
key derivation:

- **Decompression bounds**: index, residual and plain-file payloads use bounded streaming
  decompression; exceeding the declared length stops immediately, and stream end, trailing
  bytes and the exact length are checked.
- **Reference-chain bound**: the real chain depth is computed iteratively at load time and
  must satisfy `depth == parent.depth + 1`, capped at 255. Image and DUP resolution are
  iterative.
- **KDF bound**: Argon2id above 512 MiB / 8 passes / 8 lanes (scrypt `log2N>18, r>8, p>4`)
  is rejected by default, with the requested parameters in the error. Only opt in with
  `--allow-expensive-kdf` after confirming the source. Unknown crypto-header flag bits are
  rejected.

`verify --level metadata` reads no payload at all and performs structural checks only.
