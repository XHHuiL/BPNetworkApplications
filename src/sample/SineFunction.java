package sample;

import sample.bp_network.RegressionBPNetwork;

import java.util.Random;
import java.util.Scanner;

public class SineFunction {
    private static double[][] get_random_samples(int len) {
        double[][] samples = new double[len][2];
        Random random = new Random();
        for (int i = 0; i < len; i++) {
            samples[i][0] = Math.PI * (2 * random.nextDouble() - 1);
            samples[i][1] = Math.sin(samples[i][0]);
        }
        return samples;
    }

    private static double signed_to_positive(double value) {
        return (value + 1) / 2;
    }

    private static double positive_to_signed(double result) {
        return 2 * result - 1;
    }

    private static double calculate_average_error(RegressionBPNetwork network, double[][] samples_c) {
        double error = 0;
        for (double[] sample : samples_c
                ) {
            network.set_sample(sample);
            error += Math.abs(positive_to_signed(network.cal_results()[0]) - positive_to_signed(sample[1]));
        }
        return error / samples_c.length;
    }

    public static void main(String[] args) {
        RegressionBPNetwork network = new RegressionBPNetwork();
        network.init(1, 10, 1, 1,
                2.0, 0.4, 0.00001, 600000);
        double[][] samples = get_random_samples(120);
        double[][] samples_c = samples.clone();
        //deal with samples
        for (double[] sample : samples_c
                )
            sample[1] = signed_to_positive(sample[1]);
        // training
        System.out.println("Training......");
        long start_t = System.currentTimeMillis();
        network.train(samples_c);
        long end_t = System.currentTimeMillis();
        System.out.printf("Spend %dms\n", end_t - start_t);
        // print average error
        System.out.println("Average error：" + calculate_average_error(network, samples_c));

        System.out.println("Testing......");
        double error;
        double sum = 0;
        int n = 100;
        double[][] test_samples = get_random_samples(n);
        int num = 0;
        for (int k = 0; k < n; k++) {
            test_samples[k][1] = signed_to_positive(test_samples[k][1]);
            network.set_sample(test_samples[k]);
            error = Math.abs(positive_to_signed(test_samples[k][1]) -
                    positive_to_signed(network.cal_results()[0]));
            sum += error;
            if (error < 0.01)
                num++;
        }
        System.out.println("Average error：" + sum / n);
        System.out.println("Correct rate at test set: " + (100 * num / (double) n) + "%");

        // manual testing
        Scanner in = new Scanner(System.in);
        double[] sample = new double[2];
        double result;
        while (true) {
            sample[0] = in.nextDouble();
            sample[1] = signed_to_positive(Math.sin(sample[0]));
            network.set_sample(sample);
            result = positive_to_signed(network.cal_results()[0]);
            System.out.println("Result: " + result);
            System.out.println("Desired value: " + positive_to_signed(sample[1]));
            System.out.println("Absolute rate: " + Math.abs(result - positive_to_signed(sample[1])));
        }
    }
}
