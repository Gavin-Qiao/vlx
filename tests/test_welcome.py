from pathlib import Path


def test_welcome_script_reads_schema_v2_config():
    script = Path("src/vserve/welcome.sh").read_text()

    assert "yaml.safe_load" in script
    assert "backends" in script
    assert "llamacpp" in script
