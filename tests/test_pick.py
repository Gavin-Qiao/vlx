"""Tests for _pick() and _pick_many() interactive menu helpers."""

from vserve.cli import _pick, _pick_many


class TestPickNonInteractive:
    """_pick falls back to numbered prompt when not interactive (CliRunner)."""

    def test_pick_returns_index(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=False)
        mocker.patch("typer.prompt", return_value="2")
        idx = _pick(["a", "b", "c"], title="Choose:")
        assert idx == 1  # 0-indexed

    def test_pick_first_item(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=False)
        mocker.patch("typer.prompt", return_value="1")
        idx = _pick(["only"], title="Choose:")
        assert idx == 0

    def test_pick_retries_on_invalid(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=False)
        mocker.patch("typer.prompt", side_effect=["bad", "99", "2"])
        idx = _pick(["a", "b", "c"], title="Choose:")
        assert idx == 1


class TestPickManyNonInteractive:
    """_pick_many falls back to comma-separated prompt when not interactive."""

    def test_pick_many_single(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=False)
        mocker.patch("typer.prompt", return_value="2")
        result = _pick_many(["a", "b", "c"], title="Select:")
        assert result == [1]

    def test_pick_many_multiple(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=False)
        mocker.patch("typer.prompt", return_value="1,3")
        result = _pick_many(["a", "b", "c"], title="Select:")
        assert result == [0, 2]

    def test_pick_many_empty(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=False)
        mocker.patch("typer.prompt", return_value="")
        result = _pick_many(["a", "b", "c"], title="Select:")
        assert result == []

    def test_pick_many_retries_on_invalid_selection(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=False)
        prompt = mocker.patch("typer.prompt", side_effect=["1 5 2", "1 2"])
        result = _pick_many(["a", "b", "c"])
        assert result == [0, 1]
        assert prompt.call_count == 2


class TestPickWithGum:
    """_pick uses gum when available and interactive."""

    def test_pick_gum_returns_index(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=True)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0, stdout="banana\n")
        idx = _pick(["apple", "banana", "cherry"], title="Pick:")
        assert idx == 1

    def test_pick_gum_cancel(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=True)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=130, stdout="")
        idx = _pick(["apple", "banana"], title="Pick:")
        assert idx is None

    def test_pick_many_gum(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=True)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0, stdout="apple\ncherry\n")
        result = _pick_many(["apple", "banana", "cherry"])
        assert result == [0, 2]

    def test_pick_many_gum_cancel(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=True)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=130, stdout="")
        result = _pick_many(["a", "b", "c"])
        assert result == []

    def test_pick_gum_passes_title_as_header(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=True)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0, stdout="a\n")
        _pick(["a", "b"], title="My title:")
        cmd = mock_run.call_args[0][0]
        assert "--header" in cmd
        assert "My title:" in cmd

    def test_pick_gum_no_match_returns_none(self, mocker):
        """gum returns text not in items list."""
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=True)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.Mock(returncode=0, stdout="unknown\n")
        idx = _pick(["a", "b"], title="Pick:")
        assert idx is None


class TestPickWithTermMenu:
    """_pick falls back to simple-term-menu when gum is not available."""

    def test_pick_term_menu(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=False)
        mock_menu_cls = mocker.patch("simple_term_menu.TerminalMenu")
        mock_menu_cls.return_value.show.return_value = 1
        idx = _pick(["a", "b", "c"], title="Choose:")
        assert idx == 1

    def test_pick_term_menu_cancel(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=False)
        mock_menu_cls = mocker.patch("simple_term_menu.TerminalMenu")
        mock_menu_cls.return_value.show.return_value = None
        idx = _pick(["a", "b"], title="Choose:")
        assert idx is None

    def test_pick_many_term_menu(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=False)
        mock_menu_cls = mocker.patch("simple_term_menu.TerminalMenu")
        mock_menu_cls.return_value.show.return_value = (0, 2)
        result = _pick_many(["a", "b", "c"])
        assert result == [0, 2]

    def test_pick_many_term_menu_single(self, mocker):
        """TerminalMenu returns int (not tuple) when only one selected."""
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=False)
        mock_menu_cls = mocker.patch("simple_term_menu.TerminalMenu")
        mock_menu_cls.return_value.show.return_value = 1
        result = _pick_many(["a", "b", "c"])
        assert result == [1]

    def test_pick_many_term_menu_cancel(self, mocker):
        mocker.patch("vserve.cli._is_interactive", return_value=True)
        mocker.patch("vserve.cli._has_gum", return_value=False)
        mock_menu_cls = mocker.patch("simple_term_menu.TerminalMenu")
        mock_menu_cls.return_value.show.return_value = None
        result = _pick_many(["a", "b", "c"])
        assert result == []


class TestPickVariants:
    """_pick_variants uses _pick_many."""

    def test_pick_variants_returns_selected(self, mocker):
        from vserve.cli import _pick_variants
        from vserve.variants import Variant

        variants = [
            Variant(label="fp8", files={"m.safetensors": 1000}),
            Variant(label="q4", files={"m.gguf": 500}),
        ]
        mocker.patch("vserve.cli._pick_many", return_value=[1])
        result = _pick_variants(variants)
        assert len(result) == 1
        assert result[0].label == "q4"

    def test_pick_variants_single_auto_selects(self, mocker):
        """Single variant is auto-selected without a picker."""
        from vserve.cli import _pick_variants
        from vserve.variants import Variant

        variants = [Variant(label="fp8", files={"m.safetensors": 1000})]
        result = _pick_variants(variants)
        assert len(result) == 1
        assert result[0].label == "fp8"

    def test_pick_variants_empty_on_cancel(self, mocker):
        from vserve.cli import _pick_variants
        from vserve.variants import Variant

        variants = [
            Variant(label="fp8", files={"m.safetensors": 1000}),
            Variant(label="q4", files={"m.gguf": 500}),
        ]
        mocker.patch("vserve.cli._pick_many", return_value=[])
        result = _pick_variants(variants)
        assert result == []
