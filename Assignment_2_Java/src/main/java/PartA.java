import smile.classification.SVM;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;

public class PartA {

    public static void main(String[] args) {
        Classifier classifier = new Classifier(); //svm classifier initialised
        try {
            BufferedReader reader = new BufferedReader(new FileReader("src/TestingDataBinary.csv"));

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

            File out = new File("TestingResultsBinary.csv"); //output file created
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
        smile.classification.Classifier<double[]> svm; //svm classifier itself, classifies double arrays

        Classifier() {
            try{
                BufferedReader reader = new BufferedReader(new FileReader("src/TrainingDataBinary.csv"));

                ArrayList<double[]> training = new ArrayList<>(); //stores training data
                ArrayList<Integer> classification = new ArrayList<>(); //stores the labels for training data
                //both lists are kept in same order

                ArrayList<String[]> fullSet = new ArrayList<>();

                String next;
                while ((next = reader.readLine()) != null) {
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

                    if (nextData[nextData.length - 1].equals("0")) { //classifier groups between 1 and -1 instead of 1 and 0
                        classification.add(-1);
                    } else classification.add(1);
                }

                double[][] trainingData = new double[training.size()][];
                int[] labels = new int[classification.size()];

                for (int i = 0; i < classification.size(); i++) {
                    labels[i] = classification.get(i);
                }

                svm = SVM.fit(training.toArray(trainingData), labels, 100, 0.3);
                //svm is trained, C is kept large to ensure minimising fitting mistakes

                int total = 0;
                int correct = 0;

                //accuracy is calculated using remaining dataset
                for (String[] testing : fullSet) {
                    double[] test = new double[testing.length - 1];

                    for (int i = 0; i < testing.length - 1; i++) {
                        test[i] = Double.parseDouble(testing[i]);
                    }

                    int out = svm.predict(test);

                    if (out == -1 && Objects.equals(testing[testing.length - 1], "0")) correct++;
                    if (out == 1 && Objects.equals(testing[testing.length - 1], "1")) correct++;
                    total++;
                }

                System.out.println("Accuracy: " + ((double) correct / (double) total));

            } catch(IOException e) {
                System.out.println("CLASSIFIER IOEXCEPTION FOUND\n" + e);
            }
        }

        //svm classifies testing data
        int classify(double[] data) {
            int prediction = svm.predict(data);
            if (prediction == -1) prediction = 0;
            return prediction;
        }
    }
}