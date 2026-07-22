from app.evals.derivations import DERIVATIONS, derive_all
from app.evals.golden import GOLDEN_DIR, load_golden


def test_every_nonnarrative_question_has_a_derivation():
    sets = load_golden(GOLDEN_DIR)
    for s in sets:
        for q in s.questions:
            if q.expected.kind != "narrative":
                assert q.id in DERIVATIONS, f"no derivation for {q.id}"


def test_golden_yaml_matches_derivations():
    """The committed YAML values must equal what the derivations compute.

    If this fails after changing make_samples.py, run:
        python -m app.evals golden --write
    """
    derived = derive_all()
    for s in load_golden(GOLDEN_DIR):
        for q in s.questions:
            if q.expected.kind == "narrative":
                continue
            exp, got = q.expected, derived[q.id]
            assert exp.kind == got.kind, q.id
            if exp.kind in ("numeric", "categorical"):
                assert exp.value == got.value, q.id
            else:
                assert (exp.direction, exp.significant, exp.method_family) == (
                    got.direction, got.significant, got.method_family), q.id
