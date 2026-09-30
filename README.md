# NFL Total Points Predictor

## Overview

The NFL Total Points Predictor is a machine learning project built with Python and PyTorch. The goal of the project is to predict the total number of points scored in an NFL game using historical game and team statistics.

This project is being built as a way to learn more about machine learning, neural networks, data preprocessing, and model evaluation while applying those concepts to a subject I am interested in.

## Project Goals

The main goal is to build and train a neural network that can take historical NFL statistics as input and estimate the expected combined score of a game.

The initial version of the project will focus on:

- Collecting and preparing historical NFL game data
- Selecting useful team and game statistics as model features
- Creating a neural network using PyTorch
- Training the model on historical NFL games
- Evaluating predictions using metrics such as Mean Absolute Error
- Saving and loading trained model weights
- Making predictions for new NFL matchups
- Comparing the neural network against a simple statistical baseline
- Avoiding data leakage by only using information available before each game

## Technology

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Project Structure

```text
nfl-total-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_preprocess.py
│   └── test_model.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py