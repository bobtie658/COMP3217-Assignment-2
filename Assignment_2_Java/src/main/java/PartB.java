import smile.classification.LogisticRegression;
import smile.util.IntSet;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;
import java.util.Random;

public class PartB {

    public static void main(String[] args) {
        Classifier classifier = new Classifier(); //classifier initialised
        try {
            BufferedReader reader = new BufferedReader(new FileReader("src/testingDataMulti.csv"));

            ArrayList<double[]> tests = new ArrayList<>(); //stores all tests

            String next;
            while ((next = reader.readLine()) != null) { //testing data read and converted to double arrays
                String[] nextSplit = next.split(",");
                double[] doubleSplit = new double[nextSplit.length];

                for (int i = 0; i < nextSplit.length; i++) {
                    doubleSplit[i] = Double.parseDouble(nextSplit[i]);
                }

                tests.add(doubleSplit);
            }
            File out = new File("TestingResultsMulti.csv"); //output file created
            if (!out.createNewFile()) {
                out.delete();
                out.createNewFile();
            }

            BufferedWriter writer = new BufferedWriter(new FileWriter(out));

            for (double[] test : tests) {
                writer.write(classifier.classify(test) + "\n"); //testing data is classified and written to output file
                writer.flush();
            }
            writer.close();

        } catch (IOException e) {
            System.out.println("TESTING EXCEPTION FOUND\n");
        }
    }

    private static class Classifier {
        LogisticRegression.Multinomial LRC; //Logistic regression classifier

        Classifier() {
            try{
                BufferedReader reader = new BufferedReader(new FileReader("src/TrainingDataMulti.csv"));

                ArrayList<double[]> training = new ArrayList<>(); //stores training data
                ArrayList<Integer> classification = new ArrayList<>(); //stores the labels for training data
                //both lists are kept in same order

                ArrayList<String[]> fullSet = new ArrayList<>();

                String next;
                while ((next = reader.readLine()) != null) {
                    String[] split = next.split(",");

                    for (String s : split) {
                        if (Objects.equals(s, "")) System.out.println("TRUE");
                    }

                    fullSet.add(next.split(",")); //training data is read
                }

                Collections.shuffle(fullSet); //dataset is shuffled prior to training

                int iterations = (int) Math.floor(fullSet.size() * 0.9f); //90% of dataset used to train, 10% for testing

                //data entries converted to double arrays
                for (int i = 0; i < iterations; i++) {

                    String[] nextData = fullSet.get(0);
                    fullSet.remove(0);

                    double[] trainingNext = new double[nextData.length - 1];

                    for(int x = 0; x < nextData.length - 1; x++) {
                        trainingNext[x] = Double.parseDouble(nextData[x]);
                    }

                    training.add(trainingNext);

                    classification.add(Integer.valueOf(nextData[nextData.length - 1]));
                }

                //ArrayLists turned into arrays for training with smile
                double[][] trainingData = new double[training.size()][];
                int[] labels = new int[classification.size()];

                for (int i = 0; i < classification.size(); i++) {
                    labels[i] = classification.get(i);
                }

                //weights are randomly instantiated for LRC creation
                double[][] weights = new double[2][];

                for (int i = 0; i < 2; i++) {
                    double[] weight = new double[training.get(0).length + 1];

                    for (int n = 0; n < weight.length; n++) {
                        weight[n] = new Random().nextFloat();
                    }

                    weights[i] = weight;
                }

                LRC = new LogisticRegression.Multinomial(weights, 0.1, 0.2, new IntSet(new int[]{0,1,2}));
                LRC.update(training.toArray(trainingData), labels);

                //LRC = LogisticRegression.multinomial(training.toArray(trainingData), labels);
                //this declaration SHOULD work, but the weights and L are not updated with the training
                //to counteract this a new regression model is created, however even with this method, the weights are not updated upon fitting
                //I have spent hours on this one problem (with no documentation :D) and am too stubborn to learn a new module to do this

                int total = 0;
                int correct = 0;

                //accuracy is calculated using remaining dataset
                for (String[] testing : fullSet) {
                    double[] test = new double[testing.length - 1];

                    for (int i = 0; i < testing.length - 1; i++) {
                        test[i] = Double.parseDouble(testing[i]);
                    }

                    int out = LRC.predict(test);

                    if (Integer.parseInt(testing[testing.length - 1]) == out) correct++;
                    total++;
                }

                System.out.println("Accuracy: " + ((double) correct / (double) total));

            } catch(IOException e) {
                System.out.println("CLASSIFIER IOEXCEPTION FOUND\n" + e);
            }
        }

        //LRC classifies testing data
        int classify(double[] data) {
            return LRC.predict(data);
        }
    }
}
