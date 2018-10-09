package sample.bp_network;

import java.util.ArrayList;

class Node {
    ArrayList<Link> in_links = new ArrayList<>();
    ArrayList<Link> out_links = new ArrayList<>();
    double bias;
    double delta_bias;
    double delta_bias_c;
    double result;

    Node() {

    }

    Node(double bias) {
        this.bias = bias;
    }

    final double cal_result() {
        result = 0;
        for (Link link : in_links
                )
            result += link.weight * link.s_node.result;
        // use sigmoidal function to calculate result and return it
        return result = 1 / (1 + Math.exp(-result - bias));
    }
}
