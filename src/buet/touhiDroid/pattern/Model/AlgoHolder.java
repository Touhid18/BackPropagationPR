package buet.touhiDroid.pattern.Model;

import java.util.Random;

/**
 * Created by touhid on 10/19/15.
 *
 * @author touhid
 */
public class AlgoHolder {

    private static final double LEARNING_RATE = 1.0;

    private int nFeatures, nClasses;
    private double[] features, firsthidden, secHidden, thirdHidden, classIdentities;
    private double[][] inpWeights, firstHdnWghts, secHdnWghts, thirdWghts;

    public AlgoHolder(int nFeatures, int nClasses) {
        this.nFeatures = nFeatures;
        this.nClasses = nClasses;

        initWeightVectors();
    }

    private void initWeightVectors() {
        features = new double[nFeatures + 1];
        firsthidden = new double[3 + 1];
        secHidden = new double[3 + 1];
        thirdHidden = new double[3 + 1];
        classIdentities = new double[nClasses + 1];

        inpWeights = new double[3 + 1][nFeatures + 1]; // hidden x features
        firstHdnWghts = new double[3 + 1][3 + 1]; // hidden x hidden
        secHdnWghts = new double[3 + 1][3 + 1]; // hidden x hidden
        thirdWghts = new double[nClasses + 1][3 + 1]; // class x hidden

        Random rand = new Random();

        for (int j = 1; j <= 3; j++)
            for (int i = 0; i <= nFeatures; i++) {
                float f = rand.nextFloat();
                inpWeights[j][i] = f - Math.floor(f);
            }

        for (int j = 1; j <= 3; j++)
            for (int i = 0; i <= 3; i++) {
                float f = rand.nextFloat();
                firstHdnWghts[j][i] = f - Math.floor(f);
            }

        for (int j = 1; j <= 3; j++)
            for (int i = 0; i <= 3; i++) {
                float f = rand.nextFloat();
                secHdnWghts[j][i] = f - Math.floor(f);
            }

        for (int j = 1; j <= nClasses; j++)
            for (int i = 0; i <= 3; i++) {
                float f = rand.nextFloat();
                thirdWghts[j][i] = f - Math.floor(f);
            }
    }

    public void trainBackPropagation(int numIterations, int noOfSamples, double trainArray[][]) {

        for (int iterations = 0; iterations < numIterations; iterations++) {
            for (int i = 0; i < noOfSamples; i++) {
                double[] patternVector = new double[nFeatures];

                System.arraycopy(trainArray[i], 0, patternVector, 0, nFeatures);

                double[] desiredOutputVector = new double[nClasses];

                for (int k = 0; k < nClasses; k++) {
                    desiredOutputVector[k] = 0;
                }

                desiredOutputVector[((int) trainArray[i][nFeatures]) - 1] = 1;

                this.train(patternVector, desiredOutputVector);
            }
        }
    }

    private double[] train(double[] pattern, double[] desiredOutput) {
        double[] output = passToNetwork(pattern);
        backPropagation(desiredOutput);

        return output;
    }

    public double[] passToNetwork(double[] pattern) {

        for (int i = 0; i < nFeatures; i++) {
            features[i + 1] = pattern[i];
        }


        features[0] = 1.0;
        firsthidden[0] = 1.0;
        secHidden[0] = 1.0;
        thirdHidden[0] = 1.0;


        for (int j = 1; j <= 3; j++) {
            firsthidden[j] = 0.0;
            for (int i = 0; i <= nFeatures; i++) {
                firsthidden[j] += inpWeights[j][i] * features[i];
            }
            firsthidden[j] = 1.0 / (1.0 + Math.exp(-firsthidden[j]));
        }

        for (int j = 1; j <= 3; j++) {
            secHidden[j] = 0.0;
            for (int i = 0; i <= 3; i++) {
                secHidden[j] += firstHdnWghts[j][i] * firsthidden[i];
            }
            secHidden[j] = 1.0 / (1.0 + Math.exp(-secHidden[j]));
        }

        for (int j = 1; j <= 3; j++) {
            thirdHidden[j] = 0.0;
            for (int i = 0; i <= 3; i++) {
                thirdHidden[j] += secHdnWghts[j][i] * secHidden[i];
            }
            thirdHidden[j] = 1.0 / (1.0 + Math.exp(-thirdHidden[j]));
        }


        for (int j = 1; j <= nClasses; j++) {
            classIdentities[j] = 0.0;
            for (int i = 0; i <= 3; i++) {
                classIdentities[j] += thirdWghts[j][i] * thirdHidden[i];
            }
            classIdentities[j] = 1.0 / (1 + 0 + Math.exp(-classIdentities[j]));
        }

        return classIdentities;
    }


    private void backPropagation(double[] desiredOutput) {

        double[] errLevel4 = new double[nClasses + 1];
        double[] errLevel3 = new double[3 + 1];
        double[] errLevel2 = new double[3 + 1];
        double[] errLevel1 = new double[3 + 1];
        double errSum = 0.0;

        for (int i = 1; i <= nClasses; i++)
            errLevel4[i] = classIdentities[i]
                    * (1.0 - classIdentities[i])
                    * (desiredOutput[i - 1] - classIdentities[i]);


        for (int i = 0; i <= 3; i++) {
            for (int j = 1; j <= nClasses; j++)
                errSum += thirdWghts[j][i] * errLevel4[j];

            errLevel3[i] = thirdHidden[i] * (1.0 - thirdHidden[i]) * errSum;
            errSum = 0.0;
        }

        for (int i = 0; i <= 3; i++) {
            for (int j = 1; j <= 3; j++)
                errSum += secHdnWghts[j][i] * errLevel3[j];

            errLevel2[i] = secHidden[i] * (1.0 - secHidden[i]) * errSum;
            errSum = 0.0;
        }

        for (int i = 0; i <= 3; i++) {
            for (int j = 1; j <= 3; j++)
                errSum += firstHdnWghts[j][i] * errLevel2[j];

            errLevel1[i] = firsthidden[i] * (1.0 - firsthidden[i]) * errSum;
            errSum = 0.0;
        }

        for (int j = 1; j <= nClasses; j++)
            for (int i = 0; i <= 3; i++)
                thirdWghts[j][i] += LEARNING_RATE * errLevel4[j] * thirdHidden[i];

        for (int j = 1; j <= 3; j++)
            for (int i = 0; i <= 3; i++)
                secHdnWghts[j][i] += LEARNING_RATE * errLevel3[j] * secHidden[i];

        for (int j = 1; j <= 3; j++)
            for (int i = 0; i <= 3; i++)
                firstHdnWghts[j][i] += LEARNING_RATE * errLevel2[j] * firsthidden[i];

        for (int j = 1; j <= 3; j++)
            for (int i = 0; i <= nFeatures; i++)
                inpWeights[j][i] += LEARNING_RATE * errLevel1[j] * features[i];
    }
}
