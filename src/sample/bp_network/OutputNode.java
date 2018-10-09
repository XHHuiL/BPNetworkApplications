package sample.bp_network;

class OutputNode extends Node {
    double desired_value;

    OutputNode(double bias) {
        super(bias);
    }

    final double cal_exp_result() {
        result = 0;
        for (Link link : in_links
                )
            result += link.weight * link.s_node.result;
        // use exp function to calculate result and return it
        return result = Math.exp(result + bias);
    }
}