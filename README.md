# Titanic Survival Prediction API

This is a FastAPI application that predicts Titanic passenger survival using a trained ML model pipeline (`titanic_pipeline.pkl`).

## Features

- Predicts survival based on passenger features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- REST API endpoint `/Passenger` to send passenger data and receive prediction
- Model pipeline loaded from a serialized scikit-learn pipeline

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Med-Amine-03/titanic-fastapi.git
cd titanic-fastapi
