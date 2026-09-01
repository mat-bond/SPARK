# SPARK regression tests

These tests are a behavior-preservation layer for refactoring.

They do not run RFdiffusion, ProteinMPNN, Boltz, AlphaFold/ColabDesign,
PyMOL, SLURM, or Conda subprocesses. External execution is mocked.

Run from the repository root:

```bash
pytest -q
```
