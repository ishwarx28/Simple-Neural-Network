package core;

import static core.RandomGenerator.random;
import java.util.Arrays;

public class DenseLayer extends Layer {
    private double[][] weight;
    private double[] bias;

    private double[] lastX;
    private double[] lastY;

    private double[][] gw;
    private double[] gb;

    // Constructor to initialize a dense layer with input size, output size, and activation function
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction) {
        super(inputSize, outputSize, activationFunction);

        // Initialize weights and biases with random values
        this.weight = new double[inputSize][outputSize];
        this.bias = random(outputSize);
        for (int i = 0; i < inputSize; ++i) {
            weight[i] = random(outputSize);
        }

        // Initialize gradient variables
        gw = new double[inputSize][outputSize];
        gb = new double[outputSize];
    }

    @Override
    public double[][] getWeights() {
        return weight;
    }

    @Override
    public double[] getBiases() {
        return bias;
    }

    @Override
    public synchronized void teach(double[] error, double learningRate) {
        double[] backwardError = new double[inputSize];

        for (int i = 0; i < outputSize; ++i) {
            double delta = error[i] * activationFunction.derivative(lastY[i]);
            for (int j = 0; j < inputSize; ++j) {
                backwardError[j] += delta * weight[j][i];
                weight[j][i] += delta * learningRate * lastX[j];
                // Accumulate gradient for weight updates
                gw[j][i] += delta * lastX[j];
            }
            bias[i] += delta * learningRate;
            // Accumulate gradient for bias updates
            gb[i] += delta;
        }

        if (previousLayer != null) {
            previousLayer.teach(backwardError, learningRate);
        }
    }

    @Override
    public void forget() {
        // Reset the gradient accumulators to zero
        for (int i = 0; i < inputSize; ++i) {
            Arrays.fill(gw[i], 0.0);
        }
        Arrays.fill(gb, 0.0);
    }

    @Override
    public synchronized void remember(double scaler) {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weight[j][i] += gw[j][i] * scaler;
                // Reset the weight gradient accumulator
                gw[j][i] = 0.0;
            }
            bias[i] += gb[i] * scaler;
            // Reset the bias gradient accumulator
            gb[i] = 0.0;
        }

        if (getNextLayer() != null) {
            getNextLayer().remember(scaler);
        }
    }

    @Override
    public double[] forward(double[] input) {
        lastX = input;
        lastY = new double[outputSize];

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                lastY[i] += input[j] * weight[j][i];
            }
            // Apply activation function to the weighted sum
            lastY[i] = activationFunction.activate(lastY[i] + bias[i]);
        }

        // Propagate the output to the next layer if it exists
        return nextLayer == null ? lastY : nextLayer.forward(lastY);
    }
}
