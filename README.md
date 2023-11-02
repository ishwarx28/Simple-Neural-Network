# Simple Neural Network in Java

![Java](https://img.shields.io/badge/Java-8%2B-green)
![License](https://img.shields.io/badge/License-MIT-blue)

A straightforward Java implementation of a neural network for educational purposes. This project provides a basic neural network architecture, including dense layers and various activation functions.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

## Introduction

This project aims to create a simple neural network in Java, suitable for educational purposes. The neural network consists of dense layers with configurable activation functions, and it can be used for tasks such as classification or regression.

## Features

- Implementation of a basic neural network architecture.
- Customizable activation functions (ReLU and Sigmoid).
- Support for training and evaluation of the neural network.

## Getting Started

### Prerequisites

- Java 8 or higher

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ishwarx28/Simple-Neural-Network.git
   ```

2. Compile the Java source files:

   ```bash
   javac *.java
   ```

## Usage

You can simply run Main.java.

```java
java Main
```

## Training

```java
NN.fit(epochs, debugEpochStep, shuffleFlag, BatchSize, inputs, targets, learningRate);
```

## Evaluation

Evalution can be perform using ```NN.evalute``` method which returns double array size of 2 containing loss at index 0 and accuracy(0.0-1.0) at index 1.

```java
double[] result = NN.evalute(inputs, targets):
```

## License

This project is licensed under the [MIT License](LICENSE).
