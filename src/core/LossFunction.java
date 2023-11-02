package core;

public enum LossFunction {

    // Enum constants for different loss functions
    MSE {

        @Override
        public double loss(double[] y_true, double[] y_pred) {
            double loss = 0.0;
            // Calculate Mean Squared Error (MSE) loss
            for (int i = 0; i < y_true.length; ++i) {
                loss += Math.pow(y_true[i] - y_pred[i], 2);
            }
            // Return the average loss
            return loss / y_true.length;
        }

        @Override
        public double[] prime(double[] y_true, double[] y_pred) {
            double[] mse_prime = new double[y_true.length];
            // Calculate the gradient of the MSE loss
            for (int i = 0; i < y_true.length; i++) {
                mse_prime[i] = 2 * (y_true[i] - y_pred[i]) / y_true.length;
            }
            return mse_prime;
        }
    },

    MH {

        @Override
        public double loss(double[] y_true, double[] y_pred) {
            double loss = 0.0;
            // Calculate Mean Absolute Error (MH) loss
            for (int i = 0; i < y_true.length; ++i) {
                loss += Math.abs(y_true[i] - y_pred[i]);
            }
            // Return the total absolute loss
            return loss;
        }

        @Override
        public double[] prime(double[] y_true, double[] y_pred) {
            double[] mh_prime = new double[y_true.length];
            // Calculate the gradient of the MH loss
            for (int i = 0; i < y_true.length; i++) {
                mh_prime[i] = y_true[i] - y_pred[i];
            }
            return mh_prime;
        }
    };

    // Abstract methods to be implemented by each loss function
    public abstract double loss(double[] y_true, double[] y_pred);
    public abstract double[] prime(double[] y_true, double[] y_pred);
}
