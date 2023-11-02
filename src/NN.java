import core.Layer;
import core.LossFunction;
import java.io.Serializable;
import java.util.List;
import static core.RandomGenerator.shuffleData;

public class NN implements Serializable {
    private final int inputSize;
    private final int outputSize;

    private List<Layer> layers = null;
    private LossFunction lossFunction;

    // Constructor to initialize the neural network with input size, output size, layers, and loss function
    public NN(int inputSize, int outputSize, List<Layer> layers, LossFunction lossFunction) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.layers = layers;
        this.lossFunction = lossFunction;

        // Link the layers together in the neural network
        linkLayers();
    }

    // Get a layer from the list of layers by its index
    public Layer getLayer(int index) {
        return layers.get(index);
    }

    // Link the layers together in the neural network to enable forward and backward propagation
    private void linkLayers() {
        if (layers.size() <= 1) {
            return;
        }

        for (int i = 0; i < layers.size(); i++) {
            if (i == 0) {
                layers.get(i).setNextLayer(layers.get(i + 1));
            } else if (i == layers.size() - 1) {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
            } else {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
                layers.get(i).setNextLayer(layers.get(i + 1));
            }
        }
    }

    // Make predictions using the neural network
    public double[] predict(double[] input) {
        return layers.get(0).forward(input);
    }

    // Train the neural network using batch gradient descent
    public void fit(int epochs, int step, boolean shuffle, int batch_size, double[][] x_train, double[][] y_train, double learning_rate) {
        int numSample = x_train.length;
        batch_size = Math.min(numSample, Math.max(1, batch_size));

        double loss = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            boolean toPrint = epoch == epochs - 1 || epoch % step == 0;

            loss = 0.0;

            if (shuffle)
                shuffleData(x_train, y_train);

            for (int batch = 0; batch < numSample; batch += batch_size) {
                int endIndex = Math.min(numSample, batch + batch_size);

                for (int i = batch; i < endIndex; ++i) {
                    double[] out = layers.get(0).forward(x_train[i]);
                    double[] error = lossFunction.prime(y_train[i], out);

                    if (toPrint)
                        loss += lossFunction.loss(y_train[i], out);

                    layers.get(layers.size() - 1).teach(error, learning_rate);
                }

                layers.get(0).remember(1 / (endIndex - batch));
            }
            if (toPrint) {
                loss /= numSample;

                System.out.printf("Epoch (%d/%d) - loss: %.9f", epoch + 1, epochs, loss);
                System.out.println();
            }
        }
    }

    // Evaluate the neural network's performance on a dataset
    public double[] evaluate(double[][] x_train, double[][] y_train) {
        double totalLoss = 0.0;
        int numSample = x_train.length;
        int correctPredictions = 0;

        for (int i = 0; i < numSample; ++i) {
            double[] out = layers.get(0).forward(x_train[i]);

            double loss = lossFunction.loss(y_train[i], out);

            if (loss <= 1e-1) {
                correctPredictions++;
            }

            totalLoss += loss;
        }

        totalLoss /= numSample;
        double accuracy = (double) correctPredictions / (double) numSample;
		return new double[]{totalLoss, accuracy};
    }
}
