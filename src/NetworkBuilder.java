
import core.ActivationFunction;
import core.Layer;
import core.LossFunction;
import java.util.ArrayList;
import java.util.List;
import core.DenseLayer;

public class NetworkBuilder {
    private final int inputSize;
    private LossFunction lossFunction;

    private List<Layer> layers;

    // Constructor to initialize the network builder with input size and loss function
    public NetworkBuilder(int inputSize, LossFunction lossFunction) {
        this.inputSize = inputSize;
        this.lossFunction = lossFunction;
        this.layers = new ArrayList<>();
    }

    // Add a dense layer to the network with a specified output size and activation function
    public NetworkBuilder addDenseLayer(int outputSize, ActivationFunction activationFunction) {
        if (layers.isEmpty()) {
            // If it's the first layer, create it with the input size
            layers.add(new DenseLayer(inputSize, outputSize, activationFunction));
        } else {
            // For subsequent layers, use the output size of the last layer as the input size
            layers.add(new DenseLayer(layers.get(layers.size() - 1).getOutputSize(), outputSize, activationFunction));
        }
        return this;
    }

    // Build the neural network based on the configured layers and loss function
    public NN build() {
        NN network = new NN(inputSize, layers.get(layers.size() - 1).getOutputSize(), layers, lossFunction);
        return network;
    }
}
