#!/usr/bin/env python3
"""Train and register the demo credit-decision classifier in MLflow."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.model import train_and_register


def main():
    settings = get_settings()
    print(f"Training model: {settings.mlflow_model_name}")
    print(f"MLflow tracking URI: {os.path.abspath(settings.mlflow_tracking_uri)}")

    info = train_and_register(settings.mlflow_tracking_uri, settings.mlflow_model_name)

    print(f"\nModel registered successfully:")
    print(f"  Name:     {info['model_name']}")
    print(f"  Version:  {info['model_version']}")
    print(f"  Run ID:   {info['run_id']}")
    print(f"  Accuracy: {info['accuracy']:.4f}")
    print(f"  URI:      {info['artifact_uri']}")


if __name__ == "__main__":
    main()
