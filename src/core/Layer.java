package core;
import java.io.Serializable;

public abstract class Layer implements Serializable {
    // Size of the input and output for this layer
    protected final int inputSize;
    protected final int outputSize;

    // Reference to the next and previous layers in the network
    protected Layer nextLayer;
    protected Layer previousLayer;

    // Activation function applied by this layer
    protected ActivationFunction activationFunction;

    // Constructor to initialize the layer with input size, output size, and activation function
    public Layer(int inputSize, int outputSize, ActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activationFunction = activationFunction;
    }

    // Getters for input and output sizes
    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    // Getters and setters for the next and previous layers in the network
    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer _nextLayer) {
        this.nextLayer = _nextLayer;
    }

    public Layer getPreviousLayer() {
        return previousLayer;
    }

    public void setPreviousLayer(Layer _previousLayer) {
        this.previousLayer = _previousLayer;
    }

    // Abstract methods to be implemented by subclasses
    public abstract double[][] getWeights();
    public abstract double[] getBiases();

    // Methods to teach, forget, and remember weights and biases
    public abstract void teach(double[] error, double learning_rate);
    public abstract void forget();
    public abstract void remember(double scaler);

    // Forward propagation method to compute the layer's output given an input
    public abstract double[] forward(double[] input);
}
