# ============================================================================
# config.py - CONFIGURATION DE L'APPLICATION
# ============================================================================

import os
from datetime import timedelta


class Config:
    """Configuration de base"""

    # ========================
    # Chemins des ressources
    # ========================
    MODEL_PATH = os.path.join("models", "final_optimized_model.pkl")
    VECTORIZER_PATH = os.path.join(
        "data", "raw", "processed", "vectorization_objects.pkl"
    )

    # ========================
    # Paramètres généraux
    # ========================
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)

    # ========================
    # Flags d'exécution
    # ========================
    DEBUG = False
    TESTING = False
    ENV = "base"


class DevelopmentConfig(Config):
    """Configuration développement"""
    DEBUG = True
    ENV = "development"


class ProductionConfig(Config):
    """Configuration production"""
    DEBUG = False
    ENV = "production"


class TestingConfig(Config):
    """Configuration tests"""
    DEBUG = True
    TESTING = True
    ENV = "testing"


# ============================================================================
# Mapping des configurations
# ============================================================================

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
