from app.metrics import ndcg_at_k, reciprocal_rank, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "c"}, k=2) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_reciprocal_rank_when_no_match():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k_perfect_order():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0


def test_ndcg_at_k_empty_relevant_set():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0
