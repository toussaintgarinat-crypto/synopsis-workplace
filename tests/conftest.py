import os

import pytest

# Le TestClient utilise l'hôte "testserver" : on l'autorise avant l'import de main.
os.environ.setdefault("ALLOWED_HOSTS", "testserver")


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """La limite de débit est en mémoire : on la remet à zéro entre les tests."""
    from lib import security
    security.RATE_LIMITER.reset()
    yield
    security.RATE_LIMITER.reset()
