package sample.bp_network;

final class Link {
    /*
     * s_node: source node, t_node: target node
     * */
    Node s_node;
    Node t_node;
    double weight;
    double delta_weight;
    double delta_weight_c;
    double change_rate;

    Link(double weight) {
        this.weight = weight;
    }

    void set_s_node(Node s_node) {
        this.s_node = s_node;
        s_node.out_links.add(this);
    }

    void set_t_node(Node t_node) {
        this.t_node = t_node;
        t_node.in_links.add(this);
    }
}
