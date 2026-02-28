import sys

import pytest


def test_package_exports():
    import gfold

    assert hasattr(gfold, "GFoldConfig")
    assert hasattr(gfold, "GFoldSolver")


def test_config_allows_basic_overrides():
    from gfold.config import GFoldConfig

    config = GFoldConfig(n=25, gravity=[0, 0, -1.62])

    assert config.solver.n == 25
    assert config.environment.gravity == [0, 0, -1.62]


def test_config_rejects_unknown_parameter():
    from gfold.config import GFoldConfig

    with pytest.raises(ValueError, match="Unknown configuration parameter"):
        GFoldConfig(unknown_parameter=123)


def test_cli_generate_mode(monkeypatch, capsys, tmp_path):
    from gfold import cli

    class DummySolver:
        def __init__(self, config):
            self.config = config

        def generate_code(self, code_dir):
            return code_dir

        def solve(self, verbose=False):
            raise AssertionError("solve should not be called in --generate mode")

    monkeypatch.setattr(cli, "GFoldSolver", DummySolver)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gfold", "--generate", "--output", str(tmp_path), "-n", "12"],
    )

    cli.main()
    output = capsys.readouterr().out

    assert "Generated code in" in output
    assert str(tmp_path / "code") in output


def test_cli_solve_mode_no_plot(monkeypatch, capsys, tmp_path):
    from gfold import cli

    class DummySolver:
        def __init__(self, config):
            self.config = config

        def generate_code(self, code_dir):
            raise AssertionError("generate_code should not be called in solve mode")

        def solve(self, verbose=False):
            assert verbose is True
            return {"final_mass": 1234.56}

    plot_calls = []

    def fake_plot_results(solution, save_path=None, show=True):
        plot_calls.append((solution, save_path, show))

    monkeypatch.setattr(cli, "GFoldSolver", DummySolver)
    monkeypatch.setattr(cli, "plot_results", fake_plot_results)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gfold", "--no-plot", "--save-plot", "--output", str(tmp_path)],
    )

    cli.main()
    output = capsys.readouterr().out

    assert "Solving G-FOLD optimization problem" in output
    assert "Final mass: 1234.56 kg" in output
    assert len(plot_calls) == 1
    _, save_path, show = plot_calls[0]
    assert save_path == str(tmp_path / "gfold_plot.png")
    assert show is False