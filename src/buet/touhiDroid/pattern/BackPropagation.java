package buet.touhiDroid.pattern;

import buet.touhiDroid.pattern.Model.AlgoHolder;
import buet.touhiDroid.pattern.utils.Lg;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.util.Scanner;

/**
 * Created by touhid on 10/19/15.
 *
 * @author touhid
 */
public class BackPropagation {


    private int nInputs, nHidden1, nHidden2, nHidden3, nOutput;
    private double[/* i */] input, hidden1, hidden2, hidden3, output;

    private double[/* j */][/* i */] weightL1,
            weightL2,
            weightL3,
            weigthL4;

    private double learningRate = 1.0;


    //*******************************************************************************************************************


    public BackPropagation(int nInput, int nHidden, int nOutput) {

        this.nInputs = nInput;
        this.nHidden1 = nHidden;
        this.nHidden2 = nHidden;
        this.nHidden3 = nHidden;
        this.nOutput = nOutput;

        input = new double[nInput + 1];
        hidden1 = new double[nHidden + 1];
        hidden2 = new double[nHidden + 1];
        hidden3 = new double[nHidden + 1];
        output = new double[nOutput + 1];

        weightL1 = new double[nHidden + 1][nInput + 1];
        weightL2 = new double[nHidden + 1][nHidden + 1];
        weightL3 = new double[nHidden + 1][nHidden + 1];
        weigthL4 = new double[nOutput + 1][nHidden + 1];


        generateRandomWeights();
    }


    //*******************************************************************************************************************


    public void setLearningRate(double lr) {
        learningRate = lr;
    }


    private void generateRandomWeights() {
        Random rand = new Random();

        for (int j = 1; j <= nHidden1; j++)
            for (int i = 0; i <= nInputs; i++) {
                float f = rand.nextFloat();
                weightL1[j][i] = f - Math.floor(f);//Math.random() - 0.5;
            }

        for (int j = 1; j <= nHidden2; j++)
            for (int i = 0; i <= nHidden1; i++) {
                float f = rand.nextFloat();
                weightL2[j][i] = f - Math.floor(f);//Math.random() - 0.5;
            }

        for (int j = 1; j <= nHidden3; j++)
            for (int i = 0; i <= nHidden2; i++) {
                float f = rand.nextFloat();
                weightL3[j][i] = f - Math.floor(f);//Math.random() - 0.5;
            }

        for (int j = 1; j <= nOutput; j++)
            for (int i = 0; i <= nHidden3; i++) {
                float f = rand.nextFloat();
                weigthL4[j][i] = f - Math.floor(f);//Math.random() - 0.5;
            }
    }


    public double[] train(double[] pattern, double[] desiredOutput) {
        double[] output = passToNetwork(pattern);
        backpropagation(desiredOutput);

        return output;
    }


    public double[] passToNetwork(double[] pattern) {

        for (int i = 0; i < nInputs; i++) {
            input[i + 1] = pattern[i];
        }


        input[0] = 1.0;
        hidden1[0] = 1.0;
        hidden2[0] = 1.0;
        hidden3[0] = 1.0;


        for (int j = 1; j <= nHidden1; j++) {
            hidden1[j] = 0.0;
            for (int i = 0; i <= nInputs; i++) {
                hidden1[j] += weightL1[j][i] * input[i];
            }
            hidden1[j] = 1.0 / (1.0 + Math.exp(-hidden1[j]));
        }

        for (int j = 1; j <= nHidden2; j++) {
            hidden2[j] = 0.0;
            for (int i = 0; i <= nHidden1; i++) {
                hidden2[j] += weightL2[j][i] * hidden1[i];
            }
            hidden2[j] = 1.0 / (1.0 + Math.exp(-hidden2[j]));
        }

        for (int j = 1; j <= nHidden3; j++) {
            hidden3[j] = 0.0;
            for (int i = 0; i <= nHidden2; i++) {
                hidden3[j] += weightL3[j][i] * hidden2[i];
            }
            hidden3[j] = 1.0 / (1.0 + Math.exp(-hidden3[j]));
        }


        for (int j = 1; j <= nOutput; j++) {
            output[j] = 0.0;
            for (int i = 0; i <= nHidden3; i++) {
                output[j] += weigthL4[j][i] * hidden3[i];
            }
            output[j] = 1.0 / (1 + 0 + Math.exp(-output[j]));
        }

        return output;
    }


    private void backpropagation(double[] desiredOutput) {

        double[] errorL4 = new double[nOutput + 1];
        double[] errorL3 = new double[nHidden3 + 1];
        double[] errorL2 = new double[nHidden2 + 1];
        double[] errorL1 = new double[nHidden1 + 1];
        double Esum = 0.0;

        for (int i = 1; i <= nOutput; i++)
            errorL4[i] = output[i] * (1.0 - output[i]) * (desiredOutput[i - 1] - output[i]);


        for (int i = 0; i <= nHidden3; i++) {
            for (int j = 1; j <= nOutput; j++)
                Esum += weigthL4[j][i] * errorL4[j];

            errorL3[i] = hidden3[i] * (1.0 - hidden3[i]) * Esum;
            Esum = 0.0;
        }

        for (int i = 0; i <= nHidden2; i++) {
            for (int j = 1; j <= nHidden3; j++)
                Esum += weightL3[j][i] * errorL3[j];

            errorL2[i] = hidden2[i] * (1.0 - hidden2[i]) * Esum;
            Esum = 0.0;
        }

        for (int i = 0; i <= nHidden1; i++) {
            for (int j = 1; j <= nHidden2; j++)
                Esum += weightL2[j][i] * errorL2[j];

            errorL1[i] = hidden1[i] * (1.0 - hidden1[i]) * Esum;
            Esum = 0.0;
        }

        for (int j = 1; j <= nOutput; j++)
            for (int i = 0; i <= nHidden3; i++)
                weigthL4[j][i] += learningRate * errorL4[j] * hidden3[i];

        for (int j = 1; j <= nHidden3; j++)
            for (int i = 0; i <= nHidden2; i++)
                weightL3[j][i] += learningRate * errorL3[j] * hidden2[i];

        for (int j = 1; j <= nHidden2; j++)
            for (int i = 0; i <= nHidden1; i++)
                weightL2[j][i] += learningRate * errorL2[j] * hidden1[i];

        for (int j = 1; j <= nHidden1; j++)
            for (int i = 0; i <= nInputs; i++)
                weightL1[j][i] += learningRate * errorL1[j] * input[i];
    }


    //******************************************************************************************************************

    public static void main(String[] args) throws FileNotFoundException {

        // Data reading section
        Lg.pl("Reading both train and test data ...");
        Scanner trainScanner = new Scanner(new File("in/Train.txt"));
        int noOfFeatures = trainScanner.nextInt();
        int noOfClass = trainScanner.nextInt();
        int noOfSamples = trainScanner.nextInt();
        double[][] trainArray = getTrainDataArray(trainScanner, noOfSamples, noOfFeatures);
        double[][] testArray = getTestDataArray(noOfSamples, noOfFeatures);
        Lg.pl("Total train data read : " + trainArray.length);
        Lg.pl("Total test data read : " + testArray.length);

        Lg.pl("Now training the Back-Propagation algorithm ...");
        AlgoHolder ah = new AlgoHolder(noOfFeatures, noOfClass);
        ah.trainBackPropagation(1000, noOfSamples, trainArray);


        // Testing section
        Lg.pl("\nTraining complete!\nNow running experiment on " + noOfSamples + " test data ...");
        int noOfRights = 0;
        for (int i = 0; i < noOfSamples; i++) {
            double[] patternVector2 = new double[noOfFeatures];

            System.arraycopy(testArray[i], 0, patternVector2, 0, noOfFeatures);

            double[] outputReturnedVector = ah.passToNetwork(patternVector2);

            double outputReturned = 0;
            for (int m = 0; m <= noOfClass; m++) {
                if ((int) Math.round(outputReturnedVector[m]) != 0)
                    outputReturned = m;
            }

            if (outputReturned == testArray[i][noOfFeatures])
                noOfRights++;
            else
                Lg.pl("Misclassified sample no. " + i
                        + ": actual class=" + testArray[i][noOfFeatures]
                        + ", but assumed class=" + outputReturned);
        }
        Lg.pl("Total Right Classifications : " + noOfRights);
        Lg.pl("Accuracy : " + ((double) noOfRights / (double) noOfSamples) * 100 + " %");

    }

    private static double[][] getTestDataArray(int noOfSamples, int noOfFeatures) throws FileNotFoundException {
        Scanner testScanner = new Scanner(new File("in/Test.txt"));
        double[][] testArray = new double[noOfSamples][noOfFeatures + 1];

        for (int i = 0; i < noOfSamples; i++) {
            for (int j = 0; j <= noOfFeatures; j++) {
                testArray[i][j] = testScanner.nextDouble();
            }
            testScanner.nextLine();
        }
        return testArray;
    }

    private static double[][] getTrainDataArray(Scanner trainScanner, int noOfSamples, int noOfFeatures) {
        double[][] trainArray = new double[noOfSamples][noOfFeatures + 1];
        for (int i = 0; i < noOfSamples; i++) {
            for (int j = 0; j <= noOfFeatures; j++) {
                trainArray[i][j] = trainScanner.nextDouble();
            }
            trainScanner.nextLine();
        }
        return trainArray;
    }
}
