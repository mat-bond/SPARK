# SPARK regression tests

These tests are a behavior-preservation layer for refactoring.

They do not run RFdiffusion, ProteinMPNN, Boltz, AlphaFold/ColabDesign,
PyMOL, SLURM, or Conda subprocesses. External execution is mocked.

Run from the repository root:

```bash
pytest -q
```

The `xfail` tests document known bugs without making CI red:

1. multiple `BREAK` tokens are not rejected;
2. ProteinMPNN single-design mode passes the wrong keyword;
3. `b_designs_from_pm` is currently ignored when generating Boltz YAML;
4. `None` analysis cutoffs are not currently treated as disabled filters.

After fixing one of these bugs, remove its `xfail` marker and keep the
test permanently as a regression test.
