import os

import pytest

# Le TestClient utilise l'hôte "testserver" : on l'autorise avant l'import de main.
os.environ.setdefault("ALLOWED_HOSTS", "testserver")


@pytest.fixture(autouse=True)
def _reset_rate_limit(monkeypatch):
    """La limite de débit et le cache des modèles sont en mémoire : remis à zéro entre les tests.
    La clé OpenRouter de l'environnement est retirée : aucun test ne doit appeler l'API réelle."""
    from lib import security, models_catalog
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    security.RATE_LIMITER.reset()
    models_catalog.vider_cache()
    yield
    security.RATE_LIMITER.reset()
    models_catalog.vider_cache()
