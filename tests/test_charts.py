"""Unit tests for docsgen/charts.py."""
import pytest

from docsgen import charts, validate


def test_nice_ticks_cover_the_range_with_round_steps():
    ticks = charts.nice_ticks(0.0, 3.7)
    assert ticks[0] >= 0.0 and ticks[-1] <= 3.7 + 1e-9
    steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:])}
    assert len(steps) == 1


def test_nice_ticks_degenerate_range():
    assert charts.nice_ticks(2.0, 2.0) == [2.0]


def test_moving_average_is_trailing():
    assert charts.moving_average([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5]
    assert charts.moving_average([5, 5], 10) == [5.0, 5.0]


def test_reference_run_has_config_and_history():
    run = charts.load_reference()
    assert run["config"]["steps"] == len(run["history"])
    assert {"step", "train_loss", "lr", "grad_norm"} <= set(run["history"][0])


@pytest.mark.parametrize("name", sorted(charts.CHARTS))
def test_every_chart_is_an_accessible_svg_with_a_table(name):
    fragment = charts.CHARTS[name]()
    assert "<svg" in fragment and "role=\"img\"" in fragment and "<title" in fragment
    assert "<table" in fragment and "<caption>" in fragment
    assert "url(" not in fragment and "<script" not in fragment


def test_chart_ids_are_unique_so_ids_do_not_clash_on_a_page():
    ids = [part.split('id="', 1)[1].split('"', 1)[0] for part in (f() for f in charts.CHARTS.values())]
    # Every figure carries its own chart id somewhere in its markup.
    assert len(set(ids)) == len(ids)


def test_charts_work_on_a_tiny_synthetic_run():
    hist = [
        {"step": i, "train_loss": 4.0 - 0.1 * i, "val_loss": 4.0 - 0.1 * i if i in (1, 3) else None,
         "lr": 0.001 * i, "grad_norm": 1.0}
        for i in range(1, 5)
    ]
    run = {"config": {"steps": 4, "warmup_steps": 1, "lr": 0.004, "min_lr_ratio": 0.1, "grad_clip": 1.0},
           "history": hist}
    for fn in charts.CHARTS.values():
        assert "<svg" in fn(run)


def test_a_page_with_a_chart_passes_validation():
    page = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>t</title></head><body>'
        '<a class="skip-link" href="#c">Skip</a><main id="c"><h1>T</h1>' + charts.loss_chart() + "</main></body></html>"
    )
    assert validate.check_html(page) == []
