package core;

import java.util.Random;

public class RandomGenerator {
    private static Random random = new Random();

    // Set the seed for random number generation
    public static void setSeed(int seed) {
        RandomGenerator.random.setSeed(seed);
    }

    // Generate a random number from a Gaussian distribution scaled by 0.01
    public static double random() {
        return random.nextGaussian() * 0.01;
    }

    // Generate an array of random numbers with a specified length
    public static double[] random(int col) {
        double[] array = new double[col];
        for (int i = 0; i < col; ++i) {
            array[i] = random();
        }
        return array;
    }

    // Shuffle the input data (X) and labels (Y)
    public static void shuffleData(double[][] X, double[][] Y) {
        int length = X.length;

        for (int i = length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);

            // Swap X[i] with X[j]
            double[] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;

            // Swap Y[i] with Y[j]
            double[] tempY = Y[i];
            Y[i] = Y[j];
            Y[j] = tempY;
        }
    }
}
