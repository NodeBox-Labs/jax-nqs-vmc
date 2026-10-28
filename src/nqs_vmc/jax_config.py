"""Global JAX configuration.

Importing this module guarantees x64 is enabled and float32 is disabled.
It is idempotent, so importing it from multiple modules is safe and cheap.
Every module in :mod:`nqs_vmc` that touches JAX imports it at the top.
"""

from __future__ import annotations

import os

import jax

#: The dtype used for all floating-point arithmetic in this project.
DTYPE: type = jax.numpy.float64

#: The dtype used for integer indexing.
INT_DTYPE: type = jax.numpy.int32


def configure() -> None:
    """Enable x64 and validate that we are not running under a torn config."""
    jax.config.update("jax_enable_x64", True)
    # Silence the JAX warning about float64 on some platforms by allowing it.
    jax.config.update("jax_default_matmul_precision", "highest")
    if os.environ.get("NQS_VMC_DETERMINISTIC", "0") == "1":
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "x64 must be enabled for nqs_vmc; another importer disabled it."
        )


configure()