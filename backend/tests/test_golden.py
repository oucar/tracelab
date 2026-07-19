from app.evals.golden import GOLDEN_DIR, REPO_ROOT, load_golden


def test_golden_sets_load_and_are_wellformed():
    sets = load_golden(GOLDEN_DIR)
    assert {s.name for s in sets} == {"taxi", "retail", "weather"}
    ids = [q.id for s in sets for q in s.questions]
    assert len(ids) == len(set(ids))
    for s in sets:
        assert (REPO_ROOT / s.csv).exists()
        assert 10 <= len(s.questions) <= 15
        kinds = {q.expected.kind for q in s.questions}
        assert "statistical" in kinds and "narrative" in kinds
        for q in s.questions:
            if q.expected.kind == "statistical":
                assert q.expected.method_family
                assert "stats" in q.tags
