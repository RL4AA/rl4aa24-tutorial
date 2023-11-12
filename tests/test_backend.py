import pytest

from src.environments import bc, dl, ea, sh


@pytest.mark.skip(reason="Random seeds are not fixed yet")
def test_seed():
    """
    Test that using a fixed seed produces reproducible initial magnet settings and
    target beams, while different seeds produce different values.
    """
    env = section.TransverseTuning(
        backend="cheetah",
        backend_args={"incoming_mode": "random", "misalignment_mode": "random"},
    )
    _, info_ref = env.reset(seed=42)
    _, info_same = env.reset(seed=42)
    _, info_diff = env.reset(seed=24)

    # Incoming beam
    assert all(info_ref["incoming"] == info_same["incoming"])
    assert all(info_ref["incoming"] != info_diff["incoming"])

    # Misalignments
    assert all(info_ref["misalignments"] == info_same["misalignments"])
    assert all(info_ref["misalignments"] != info_diff["misalignments"])
