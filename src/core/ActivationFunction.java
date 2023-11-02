package core;

public enum ActivationFunction {

    RELU {
        @Override
        public double activate(double x) {
            // ReLU (Rectified Linear Unit) activation function
            return Math.max(0, x);
        }

        @Override
        public double derivative(double x) {
            // Derivative of the ReLU activation function
            return x <= 0 ? leak : 1;
        }
    },

    SIGMOID {
        @Override
        public double activate(double x) {
            // Sigmoid activation function
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double derivative(double x) {
            // Derivative of the Sigmoid activation function
            return x * (1 - x);
        }
    };

    // Common parameter used in the ReLU derivative
    private static double leak = 0.01;

    // Abstract methods to be implemented by each activation function
    public abstract double activate(double x);
    public abstract double derivative(double x);
}
