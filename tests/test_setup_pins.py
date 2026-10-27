from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGPO_PIN = "87e4061c998ebb3a348074a6f70b4599fb9f373e"


def test_igpo_installer_defaults_to_the_audited_commit():
    installer = (ROOT / "integrations/igpo/install.sh").read_text(encoding="utf-8")
    assert f'BASE="${{VDRA_IGPO_BASE:-{IGPO_PIN}}}"' in installer
