M6 / E7 LOCAL PATCH — NOT PUSHED TO GITHUB

Files to copy:
  src/rulfl/eval/stats.py
  tests/test_eval_stats.py

Changes addressing teammate feedback:
1. Floating-point noise: prior to signed-rank calculation, differences are
   rounded to 10 decimal places (RUL-cycle units). Near-zero noise therefore
   becomes zero and near-equal magnitudes share midranks. Exact enumeration
   uses integer doubled ranks to avoid floating-point tail-comparison errors.
2. Checked E7 in docs/DECISIONS.md from the supplied repository archive:
   E7 REQUIRES: paired comparisons by (H, seed), mean differences vs FedAvg,
   95% bootstrap interval over seeds, one-sided exact Wilcoxon per H and
   pooled H>=0.5 (15 pairs), NumPy-only implementation.
   E7 DOES NOT SPECIFY: percentile versus BCa bootstrap, explicit 'two-sided'
   bootstrap nomenclature, zero-difference policy, rounding tolerance, or
   the maximum sample count. Implementation currently chooses percentile
   bounds at 2.5% and 97.5% (two-sided 95% interval), Wilcox drop-zeros,
   a 10-decimal precision convention, and a protective cap of 20 INPUT pairs.
   These choices should be approved/documented before registered experiments
   if the team treats them as frozen protocol decisions.
3. Only .py and this README are packaged: no __pycache__ directories or .pyc.

Directions:
  Copy files into matching paths of your CURRENT local checkout.
  python -m pytest -q tests/test_eval_stats.py
  python -m pytest -q

The optional SciPy comparison in one test uses pytest.importorskip; the
production implementation uses only numpy, and SciPy is not a dependency.

No GitHub action was performed. No experiment result is claimed.
