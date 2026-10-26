# Vendored reference implementations

`upstream_defense.py` and `upstream_label_consistency.py` are unmodified
copies of `v02/1_scripts/00_common/{defense,label_consistency}.py` from
[Single_Cell_Poisoning](https://github.com/Georgakopoulos-Soares-lab/Single_Cell_Poisoning),
MIT licensed (see `UPSTREAM_LICENSE`).

They are **test fixtures, not dependencies**: nothing in `src/` imports
them. `test_vote_equivalence.py` and `test_label_consistency_equivalence.py`
run them side by side with the ported implementations on random inputs, so
a change that quietly alters the semantics of a ported layer fails CI.

`schemas.py` is a two-line shim supplying the one constant the upstream
module imports.
