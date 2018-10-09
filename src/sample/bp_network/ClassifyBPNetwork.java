package sample.bp_network;

import java.util.Random;

public class ClassifyBPNetwork {
    private InputNode[] in_layer;
    private Node[][] hide_layers;
    private OutputNode[] out_layer;
    private double learning_rate;
    private int num_in;
    private int num_hide;
    private int num_hide_layer;
    private int batch_size;
    private int num_out;
    private double threshold;
    private double weight_decay;
    private double pass_rate;
    private int num_epoch;

    public void init(int num_in, int num_hide, int num_hide_layer, int num_out, int batch_size,
                     double learning_rate, double threshold, double weight_decay, double pass_rate, int num_epoch) {
        in_layer = new InputNode[num_in];
        hide_layers = new Node[num_hide_layer][num_hide];
        out_layer = new OutputNode[num_out];
        this.learning_rate = learning_rate;
        this.num_in = num_in;
        this.num_hide = num_hide;
        this.num_hide_layer = num_hide_layer;
        this.batch_size = batch_size;
        this.num_out = num_out;
        this.threshold = threshold;
        this.weight_decay = weight_decay;
        this.pass_rate = pass_rate;
        this.num_epoch = num_epoch;
        Random random = new Random();
        for (int i = 0; i < num_in; i++)
            in_layer[i] = new InputNode();
        for (int i = 0; i < num_hide_layer; i++) {
            hide_layers[i] = new Node[num_hide];
            for (int j = 0; j < num_hide; j++)
                hide_layers[i][j] = new Node(random.nextGaussian() / 2 - 0.5);
        }
        for (int i = 0; i < num_out; i++)
            out_layer[i] = new OutputNode(random.nextGaussian() * 0.2);
        // build links between hide layer and input layer
        for (int j = 0; j < num_hide; j++) {
            for (int i = 0; i < num_in; i++) {
                Link link = new Link(random.nextGaussian() / Math.sqrt(num_in));
                link.set_s_node(in_layer[i]);
                link.set_t_node(hide_layers[num_hide_layer - 1][j]);
            }
        }
        // build links between hide layers
        for (int i = 0; i < num_hide_layer - 1; i++)
            for (int k = 0; k < num_hide; k++
                    )
                for (int j = 0; j < num_hide; j++) {
                    Link link = new Link(random.nextGaussian() / Math.sqrt(num_hide));
                    link.set_s_node(hide_layers[i + 1][j]);
                    link.set_t_node(hide_layers[i][k]);
                }
        // build links between output layer and hide layer
        for (int j = 0; j < num_out; j++) {
            for (int i = 0; i < num_hide; i++) {
                Link link = new Link(random.nextGaussian() / Math.sqrt(num_hide));
                link.set_s_node(hide_layers[0][i]);
                link.set_t_node(out_layer[j]);
            }
        }
    }

    final public void set_sample(double[] sample) {
        for (int i = 0; i < num_in; i++)
            in_layer[i].result = sample[i];
        for (int i = 0; i < num_out; i++)
            out_layer[i].desired_value = sample[num_in + i];
    }

    final public double[] cal_results() {
        for (int i = num_hide_layer - 1; i >= 0; i--)
            for (int j = 0; j < num_hide; j++)
                hide_layers[i][j].cal_result();
        double[] results = new double[num_out];
        double temp = 0;
        for (int i = 0; i < num_out; i++)
            temp += out_layer[i].cal_exp_result();
        // transform results to probability distributions
        for (int i = 0; i < num_out; i++) {
            out_layer[i].result = out_layer[i].result / temp;
            results[i] = out_layer[i].result;
        }
        // store delta weight and delta bias
        for (int i = 0; i < num_out; i++) {
            out_layer[i].delta_bias_c += out_layer[i].result - out_layer[i].desired_value;
            for (Link link : out_layer[i].in_links
                    )
                link.delta_weight_c += link.s_node.result * (out_layer[i].result - out_layer[i].desired_value);
        }
        return results;
    }

    final public double[] get_results() {
        double[] results = new double[num_out];
        for (int i = 0; i < num_out; i++)
            results[i] = out_layer[i].result;
        return results;
    }

    final public double[] desired_values() {
        double[] desired_values = new double[num_out];
        for (int i = 0; i < num_out; i++)
            desired_values[i] = out_layer[i].desired_value;
        return desired_values;
    }

    private double get_error() {
        double result = 0;
        for (int i = 0; i < out_layer.length; i++) {
            result += out_layer[i].desired_value * Math.log(out_layer[i].result);
        }
        return -result;
    }

    public void train(double[][] samples) {
        int len = samples.length;
        int num_of_epoch = 0;
        boolean can_stop = false;
        int success_num;
        double error;
        double average_error;
        while (!can_stop && num_of_epoch < num_epoch) {
            num_of_epoch++;
            success_num = 0;
            error = 0;
            average_error = 0;
            for (int i = 0; i < len; i++) {
                set_sample(samples[i]);// process input to meet bp network
                cal_results();
                error += get_error();
                average_error += get_error();
                if ((i + 1) % batch_size == 0) {
                    adjust();
                    // clean
                    for (int j = 0; j < num_out; j++) {
                        out_layer[j].delta_bias_c = 0;
                        for (Link link : out_layer[j].in_links
                                )
                            link.delta_weight_c = 0;
                    }
                    if (error / batch_size < threshold)// compare results and desired values
                        success_num += batch_size;
                    error = 0;
                }
            }
            average_error /= len;
            // print information
            if (success_num / (double) len >= pass_rate) {
                can_stop = true;
                System.out.println("Epoch " + num_of_epoch + ": over desire pass rate!"
                        + " average error:" + average_error + " learning rate: " + learning_rate);
            } else
                System.out.println("Epoch " + num_of_epoch + String.format(":%.1f", success_num / (double) len * 100)
                        + "% average error: " + average_error + " learning rate: " + learning_rate);
            if (num_of_epoch >= 1)
                learning_rate = 0.01;
        }
    }

    private void adjust() {
        double dt;
        // calculate delta weights and adjust biases
        for (int i = 0; i < num_out; i++) {
            out_layer[i].delta_bias = -learning_rate *
                    (out_layer[i].delta_bias_c / batch_size + weight_decay * out_layer[i].bias);
            out_layer[i].bias += out_layer[i].delta_bias;
            for (Link link : out_layer[i].in_links
                    )
                link.change_rate = link.delta_weight_c / batch_size + weight_decay * link.weight;
        }
        for (int i = 0; i < num_hide_layer; i++) {
            for (int j = 0; j < num_hide; j++) {
                dt = 0;
                for (Link high_l : hide_layers[i][j].out_links
                        )
                    dt += high_l.weight * high_l.change_rate;
                dt *= (1 - hide_layers[i][j].result);
                hide_layers[i][j].delta_bias = -learning_rate * dt;
                hide_layers[i][j].bias += hide_layers[i][j].delta_bias;
                for (Link low_l : hide_layers[i][j].in_links
                        )
                    low_l.change_rate = dt * low_l.s_node.result;
            }
        }
        // update weights
        for (int i = 0; i < num_out; i++)
            for (Link link : out_layer[i].in_links
                    ) {
                link.delta_weight = -learning_rate * link.change_rate;
                link.weight += link.delta_weight;
            }
        for (int i = 0; i < num_hide_layer; i++)
            for (int j = 0; j < num_hide; j++)
                for (Link link : hide_layers[i][j].in_links
                        ) {
                    link.delta_weight = -learning_rate * link.change_rate;
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
