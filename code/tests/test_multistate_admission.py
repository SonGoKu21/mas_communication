import pytest
from scripts.prepare_shopping_multistate_tasks import task_variants, assert_transition


def test_variants_have_distinct_ids_and_nontrivial_increases_decreases():
    product = {"task_id": "shopping-001", "product_title": "Tea", "product_url": "http://localhost:17770/tea", "quantity": 1}
    variants = task_variants(product, 0)
    assert len(variants) == 2
    assert len({t["task_id"] for t in variants}) == 2
    assert variants[0]["initial_quantity"] < variants[0]["quantity"]
    assert variants[1]["initial_quantity"] > variants[1]["quantity"]
    assert product["quantity"] == 1


def test_transition_checks_real_observed_quantity_not_just_true_flag():
    t = {"task_id": "t", "product_title": "Tea", "quantity": 2}
    good = {"task_id": "t", "product_title": "Tea", "observed_quantity": 2,
            "cart_verified": True, "sku": "TEA", "product_id": "1"}
    assert_transition(t, good)
    with pytest.raises(ValueError):
        assert_transition(t, {**good, "observed_quantity": 3})
