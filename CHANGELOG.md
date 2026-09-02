# Changelog

All notable changes to SPARK are documented in this file.

## Unreleased

### Fixed

- ProteinMPNN single-design execution no longer passes an invalid keyword argument.
- `b_designs_from_pm` is now respected when generating Boltz YAML inputs.
- Optional pLDDT, PAE, and PDE cutoffs can be disabled by leaving them unset.
- Multiple `BREAK` markers in a single input chain are rejected.

### Changed

- Reorganized implementation under `src/` while retaining the root
  `pipeline.py` compatibility entry point.

### Added

- Regression test suite for core SPARK-owned pipeline logic.
