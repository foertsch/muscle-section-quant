from dataclasses import asdict

from msq.core.problems import ExitStatus, Problem, errors


def test_problem_text_and_round_trip():
    p = Problem("warning", "only 1 control animal", "TA", code="few_controls")
    assert str(p) == "WARNING: TA: only 1 control animal"
    assert str(Problem("error", "the sample table is empty")) == "ERROR: the sample table is empty"
    assert Problem(**asdict(p)) == p
    old = Problem(**{"level": "error", "message": "no controls.", "where": "TA"})  # written before codes existed
    assert old.code == "" and str(old) == "ERROR: TA: no controls."


def test_errors_filters_by_level():
    ps = [Problem("error", "a"), Problem("warning", "b"), Problem("note", "c"), Problem("error", "d")]
    assert [p.message for p in errors(ps)] == ["a", "d"]


def test_exit_statuses_are_the_runner_contract():
    assert [int(s) for s in ExitStatus] == [0, 1, 2, 3, 4, 5]
    assert ExitStatus.LOCKED == 5 and ExitStatus.DONE == 0
