from analysis.check_leakage import norm
import pytest

from analysis.datasets_util import _gold, _id, _problem


def test_norm_keeps_semantic_math_commands_and_signs():
    assert norm(r"\sin 30^\circ") != norm(r"\tan 30^\circ")
    assert norm(r"z^4 + z^2 + 1") != norm(r"z^4 - z^2 + 1")


def test_norm_keeps_expression_grouping():
    assert norm("a/(b+c)") != norm("a/b+c")


def test_norm_canonicalizes_formatting_and_simple_fraction_notation():
    assert norm(r"\sin ^{2} 4^{\circ} + \cdots + \sin ^{2} 8^{\circ}") == norm(
        r"$\sin^2 4^\circ + \dots + \sin^2 8^\circ$"
    )
    assert norm(r"a+\frac{1}{b}") == norm("a+(1/b)")


def test_norm_preserves_case_sensitive_math_symbols():
    assert norm("A + B") != norm("a + b")


def test_nested_training_export_fields_preserve_original_problem_id():
    row = {"extra_info": {"problem": "original text", "index": 391}}
    assert _problem(row) == "original text"
    assert _id(row, 7) == "391"


def test_evaluation_gold_answer_must_be_present_or_extractable():
    assert _gold({"answer": "42"}) == "42"
    assert _gold({"answer": [0]}) == "0"
    assert _gold({"answer": [None], "final_answer": "42"}) == "42"
    assert _gold({"solution": r"The answer is \boxed{42}."}) == "42"
    with pytest.raises(KeyError, match="no gold answer"):
        _gold({"problem": "unlabeled question"})
    with pytest.raises(KeyError, match="no gold answer"):
        _gold({"answer": "", "solution": "no boxed answer"})
    with pytest.raises(KeyError, match="no gold answer"):
        _gold({"answer": [None], "solution": "no boxed answer"})
    with pytest.raises(KeyError, match="no gold answer"):
        _gold({"answer": ["  "], "solution": "no boxed answer"})
