"""Opt-in launcher hook for the exact world-size-one collective workaround."""

from __future__ import annotations

import os


if os.environ.get("VDRA_SINGLE_RANK_COLLECTIVES") == "1":
    from modal_app.single_rank_collectives import install_single_rank_collective_identity

    install_single_rank_collective_identity()
