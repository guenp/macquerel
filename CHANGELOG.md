# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-02

### Added

- Initial release: quantum state-vector simulator for Apple Silicon.
- `Circuit` builder with a chainable gate API and `Simulator` with `run()` / `statevector()`.
- CPU backend (NumPy reference, tensordot reshape).
- MLX backend for Apple Silicon GPU acceleration (17–30 qubits), with graceful fallback.
- Metal backend (PyObjC driver, 64-bit indexing, in-place updates) reaching 31–33 qubits past
  MLX's int32 ceiling.
- Automatic backend selection (CPU ≤16q, MLX 17–30q, Metal 31q+).
- Gate-fusion compiler and diagonal/permutation/dense gate classification.
- Cirq and Qiskit import adapters (optional extras).

[Unreleased]: https://github.com/guenp/macquerel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/guenp/macquerel/releases/tag/v0.1.0
