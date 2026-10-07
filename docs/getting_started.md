# Getting Started

AVF currently contains the framework skeleton and the migrated TestForge venture.

1. Create a Python 3.11 or newer environment.
2. Install the AVF project dependencies.
3. Install venture-specific dependencies from `ventures/testforge/requirements.txt` as needed.
4. Copy `.env.example` to `.env` and add local credentials only when running workflows that require external models.

CI and compile checks must not require real API keys.
