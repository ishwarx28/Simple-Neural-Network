import java.util.*;
import core.LossFunction;
import core.ActivationFunction;
import core.RandomGenerator;

public class Main {
    public static void main(String[] args) {
        RandomGenerator.setSeed(222);

        // Define input and target data
        double[][] inputs = {
            {1.0, 0.0},
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
        };

        double[][] targets = {
            {1.0},
            {0.0},
            {1.0},
            {0.0}
        };

        // Create a neural network with specified architecture and loss function
        NN nn = new NetworkBuilder(inputs[0].length, LossFunction.MH)
            .addDenseLayer(10, ActivationFunction.RELU)
            .addDenseLayer(1, ActivationFunction.SIGMOID)
            .build();

        System.out.println("Training...");

        // Training configuration
        int debugStep = 1000;
        int epochs = 5000;
        boolean shuffle = true;
        int batch_size = 4;
        double learningRate = 0.1;

        // Train the neural network
        nn.fit(epochs, debugStep, shuffle, batch_size, inputs, targets, learningRate);

        // Evaluate the trained network
        System.out.println();
        System.out.println("Evaluating...");
        double[] result = nn.evaluate(inputs, targets);
        System.out.println("Loss: " + result[0]);
        System.out.println("Accuracy: " + (100 * result[1]) + "%");

        // Test the network with new data
        System.out.println();
        System.out.println("Testing unseen data...");
        double[][] newInputs = {
            {0.5, 0.5},
            {0.0, 0.7},
            {0.8, 0.8}
        };

        for (double[] input : newInputs) {
            double[] output = nn.predict(input);

            System.out.println("Input: " + Arrays.toString(input));
            System.out.println("Predicted output: " + Arrays.toString(output));
        }
    }
}
