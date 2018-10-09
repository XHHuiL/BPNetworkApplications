package sample.bp_network;

import java.util.Random;

public class RegressionBPNetwork {
    private InputNode[] in_layer;
    private Node[][] hide_layers;
    private OutputNode[] out_layer;
    private double learning_rate;
    private double damp_rate;
    private int num_in;
    private int num_hide;
    private int num_hide_layer;
    private int num_out;
    private double threshold;
    private int num_epoch;

    public void init(int num_in, int num_hide, int num_hide_layer, int num_out,
                     double learning_rate, double damp_rate, double threshold, int num_epoch) {
        in_layer = new InputNode[num_in];
        hide_layers = new Node[num_hide_layer][num_hide];
        out_layer = new OutputNode[num_out];
        this.learning_rate = learning_rate;
        this.damp_rate = damp_rate;
        this.num_in = num_in;
        this.num_hide = num_hide;
        this.num_hide_layer = num_hide_layer;
        this.num_out = num_out;
        this.threshold = threshold;
        this.num_epoch = num_epoch;
        Random random = new Random();
        for (int i = 0; i < num_in; i++)
            in_layer[i] = new InputNode();
        for (int i = 0; i < num_hide_layer; i++) {
            hide_layers[i] = new Node[num_hide];
            for (int j = 0; j < num_hide; j++)
                hide_layers[i][j] = new Node(-random.nextGaussian());
        }
        for (int i = 0; i < num_out; i++)
            out_layer[i] = new OutputNode(-random.nextGaussian());
        // build links between hide layer and input layer
        for (Node hide_node : hide_layers[num_hide_layer - 1]
                ) {
            for (int i = 0; i < num_in; i++) {
                Link link = new Link(random.nextGaussian());
                link.set_s_node(in_layer[i]);
                link.set_t_node(hide_node);
            }
        }
        // build links between hide layers
        for (int i = 0; i < num_hide_layer - 1; i++)
            for (Node hide_node : hide_layers[i]
                    )
                for (int j = 0; j < num_hide; j++) {
                    Link link = new Link(random.nextGaussian());
                    link.set_s_node(hide_layers[i + 1][j]);
                    link.set_t_node(hide_node);
                }
        // build links between output layer and hide layer
        for (Node output_node : out_layer
                ) {
            for (int i = 0; i < num_hide; i++) {
                Link link = new Link(random.nextGaussian());
                link.set_s_node(hide_layers[0][i]);
                link.set_t_node(output_node);
            }
        }
    }

    public void set_sample(double[] sample) {
        for (int i = 0; i < num_in; i++)
            in_layer[i].result = sample[i];
        for (int i = 0; i < num_out; i++)
            out_layer[i].desired_value = sample[num_in + i];
    }

    public double[] cal_results() {
        for (int i = num_hide_layer - 1; i >= 0; i--)
            for (Node node : hide_layers[i]
                    )
                node.cal_result();
        double[] results = new double[num_out];
        for (int i = 0; i < num_out; i++)
            results[i] = out_layer[i].cal_result();
        return results;
    }

    public double[] get_results() {
        double[] results = new double[num_out];
        for (int i = 0; i < num_out; i++)
            results[i] = out_layer[i].result;
        return results;
    }

    private double[] desired_values() {
        double[] desired_values = new double[num_out];
        for (int i = 0; i < num_out; i++)
            desired_values[i] = out_layer[i].desired_value;
        return desired_values;
    }

    private static double get_error(double[] results, double[] desired_values) {
        double result = 0;
        for (int i = 0; i < results.length; i++) {
            result += Math.pow(desired_values[i] - results[i], 2);
        }
        return result / 2;
    }

    public void train(double[][] samples) {
        StringBuilder stringBuilder = new StringBuilder();
        int len = samples.length;
        int num_of_epoch = 0;
        boolean can_stop = false;
        int success_num;
        while (!can_stop) {
            num_of_epoch++;
            success_num = 0;
            for (double[] sample : samples) {
                set_sample(sample);// process input to meet bp network
                if (get_error(cal_results(), desired_values()) < threshold) {
                    // compare results and desired values
                    success_num++;
                } else
                    adjust();
            }
            if (success_num == len) {
                can_stop = true;
                System.out.print(stringBuilder.toString());
                System.out.println("Epoch " + num_of_epoch + ": 100%!");
            } else {
                stringBuilder.append("Epoch ").append(num_of_epoch).append(": ")
                        .append(String.format("%.1f", success_num / (double) len * 100)).append("%\n");
                if (num_of_epoch % 40 == 0) {
                    System.out.println(stringBuilder.toString());
                    stringBuilder.delete(0, stringBuilder.length());
                }
            }
            if (num_of_epoch > num_epoch) {
                System.out.println("over num_epoch");
                can_stop = true;
            }
        }
    }

    private void adjust() {
        double dt, di, oi;
        // calculate delta weights and adjust biases
        for (OutputNode out_nd : out_layer
                ) {
            di = out_nd.desired_value;
            oi = out_nd.result;
            dt = oi * (1 - oi) * (oi - di);
            out_nd.delta_bias = -learning_rate * dt + damp_rate * out_nd.delta_bias;
            out_nd.bias += out_nd.delta_bias;
            for (Link link : out_nd.in_links
                    )
                link.change_rate = dt * link.s_node.result;
        }
        for (Node[] hide_layer : hide_layers
                ) {
            for (Node hide_nd : hide_layer
                    ) {
                dt = 0;
                for (Link high_l : hide_nd.out_links
                        )
                    dt += high_l.weight * high_l.change_rate;
                dt *= (1 - hide_nd.result);
                hide_nd.delta_bias = -learning_rate * dt + damp_rate * hide_nd.delta_bias;
                hide_nd.bias += hide_nd.delta_bias;
                for (Link low_l : hide_nd.in_links
                        )
                    low_l.change_rate = dt * low_l.s_node.result;
            }
        }
        // update weights
        for (OutputNode node : out_layer
                )
            for (Link link : node.in_links
                    ) {
                link.delta_weight = -learning_rate * link.change_rate + damp_rate * link.delta_weight;
                link.weight += link.delta_weight;
            }
        for (Node[] hide_layer : hide_layers
                )
            for (Node node : hide_layer
                    )
                for (Link link : node.in_links
                        ) {
                    link.delta_weight = -learning_rate * link.change_rate + damp_rate * link.delta_weight;
                    link.weight += link.delta_weight;
                }
    }

    public void print_weight() {
        System.out.println("-----------------Output Layer-----------------");
        for (OutputNode node : out_layer
                ) {
            System.out.print("[Weight: ");
            for (Link link : node.in_links
                    ) {
                System.out.printf("%.6f,", link.weight);
            }
            System.out.printf("Bias: %.6f]\n", node.bias);
        }
        System.out.println("-------------------Hide Layer-------------------");
        for (int i = 0; i < num_hide_layer; i++) {
            System.out.println("Hide Layer: " + (i + 1) + "----------------------");
            for (Node node : hide_layers[i]
                    ) {
                System.out.print("[Weight: ");
                for (Link link : node.in_links
                        ) {
                    System.out.printf("%.6f,", link.weight);
                }
                System.out.printf("Bias: %.6f]\n", node.bias);
            }
        }
    }
}
