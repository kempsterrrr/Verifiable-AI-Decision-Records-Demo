"""Setup for ario-mlflow plugin."""

from setuptools import setup, find_packages

setup(
    name="ario-mlflow",
    version="0.1.0",
    description="ar.io MLflow plugin — verifiable provenance for the ML lifecycle",
    packages=find_packages(include=["ario_mlflow", "ario_mlflow.*"]),
    python_requires=">=3.10",
    install_requires=[
        "mlflow>=2.14.0",  # mlflow.trace / set_trace_tag / get_active_trace_id / get_trace APIs
        "PyNaCl>=1.5.0",
        "turbo-sdk>=0.0.5",
        "requests>=2.31.0",
    ],
    entry_points={
        "mlflow.run_context_provider": [
            "ario=ario_mlflow.plugin:ArioContextProvider",
        ],
        "console_scripts": [
            "ario-mlflow=ario_mlflow.cli:main",
        ],
    },
)
