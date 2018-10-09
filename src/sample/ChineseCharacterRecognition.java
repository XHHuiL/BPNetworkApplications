package sample;

import sample.bp_network.ClassifyBPNetwork;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;

public class ChineseCharacterRecognition {
    private static String chars = "苟利国家生死以岂因祸福避趋之";
    private static int size = 28;
    private static int num_of_input = size * size;
    private static int num_of_category = 14;
    private static int num_of_train_group = 205;
    private static int num_of_test_group = 256 - num_of_train_group;

    private static double[][] get_samples(int num, int offset) throws IOException {
        double[][] samples = new double[num_of_category * num][num_of_input + num_of_category];
        BufferedImage image;
        int index;
        for (int m = 0; m < num; m++) {
            for (int i = 1; i <= num_of_category; i++) {
                index = m * num_of_category + i - 1;
                image = ImageIO.read(new File("train_set/" + i + "/" + (m + offset) + ".bmp"));
                for (int j = 0; j < size; j++)
                    for (int k = 0; k < size; k++)
                        samples[index][j * size + k] = (image.getRGB(k, j) & 0xffffff) == 0xffffff ? 0 : 1;
                samples[index][num_of_input + i - 1] = 1;
            }
        }
        return samples;
    }

    private static double[][] get_foo_test_samples() throws IOException {
        double[][] samples = new double[num_of_category][num_of_input + num_of_category];
        BufferedImage image;
        for (int i = 1; i <= num_of_category; i++) {
            image = ImageIO.read(new File("test_set/" + i + ".PNG"));
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; k++)
                    samples[i - 1][j * size + k] = (image.getRGB(k, j) & 0xffffff) == 0xffffff ? 0 : 1;
            samples[i - 1][num_of_input + i - 1] = 1;
        }
        return samples;
    }

    private static double[] num_to_vector(double[] sample) {
        int len = sample.length;
        double[] new_sample = new double[num_of_input + num_of_category];
        System.arraycopy(sample, 0, new_sample, 0, len - 1);
        new_sample[len - 2 + (int) sample[len - 1]] = 1;
        return new_sample;
    }

    // print result
    private static char get_char(double[] results) {
        int index = 0;
        double value = 0;
        for (int i = 0; i < results.length; i++)
            if (value <= results[i]) {
                value = results[i];
                index = i;
            }
        return chars.charAt(index);
    }

    public static void main(String[] args) throws IOException {
        ClassifyBPNetwork network = new ClassifyBPNetwork();
        System.out.println("Load images......");
        long start_t = System.currentTimeMillis();
        double[][] samples = get_samples(num_of_train_group, 0);
        long end_t = System.currentTimeMillis();
        System.out.printf("Spend %ds\n", (end_t - start_t) / 1000);

        // training
        System.out.println("Train......");
        network.init(num_of_input, 150, 1, num_of_category, 1,
                0.2, 0.5, 0, 0.99, 14);
        start_t = System.currentTimeMillis();
        network.train(samples);
        end_t = System.currentTimeMillis();
        System.out.printf("Spend %ds\n", (end_t - start_t) / 1000);

        //testing in train set
        int suc_num = 0;
        for (int i = 0; i < samples.length; i++) {
            network.set_sample(samples[i]);
            char result = get_char(network.cal_results());
            char desire = get_char(network.desired_values());
            if (result == desire)
                suc_num++;
        }
        System.out.println("Correct rate at train set:");
        System.out.println(suc_num / (double) samples.length);

        // testing in test set
        suc_num = 0;
        double[][] test_samples = get_samples(num_of_test_group, num_of_train_group);
        System.out.println("Correct rate at test set:");
        for (int i = 0; i < test_samples.length; i++) {
            network.set_sample(test_samples[i]);
            char result = get_char(network.cal_results());
            char desire = get_char(network.desired_values());
            if (result == desire)
                suc_num++;
        }
        System.out.println(suc_num / (double) test_samples.length);

        // manual test
        Scanner in = new Scanner(System.in);
        String[] strings;
        double[] sample = new double[num_of_input + 1];
        while (true) {
            strings = in.nextLine().split(" ");
            BufferedImage image = ImageIO.read(new File(strings[0]));
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; k++)
                    sample[j * size + k] = (image.getRGB(k, j) & 0xffffff) == 0xffffff ? 0 : 1;
            sample[num_of_input] = Double.parseDouble(strings[1]);
            network.set_sample(num_to_vector(sample));
            System.out.println(chars.charAt((int) Double.parseDouble(strings[1]) - 1)
                    + ":" + get_char(network.cal_results()));
        }
    }
}
