package ua.coursework.regression.benchmark;

import ua.coursework.regression.model.DataPoint;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DataGenerator {

    private final Random random;

    public DataGenerator() {
        this.random = new Random(42);
    }

    public DataGenerator(long seed) {
        this.random = new Random(seed);
    }

    public List<DataPoint> generateMultipleRegressionData(int size, double[] coefficients,
            double noiseLevel) {
        int n = coefficients.length - 1;
        List<DataPoint> dataPoints = new ArrayList<>(size);

        for (int i = 0; i < size; i++) {
            double[] x = new double[n];
            double y = coefficients[0];

            for (int j = 0; j < n; j++) {
                x[j] = random.nextDouble() * 10;
                y += coefficients[j + 1] * x[j];
            }

            y += random.nextGaussian() * y * noiseLevel;
            dataPoints.add(new DataPoint(x, y));
        }

        return dataPoints;
    }

    public List<DataPoint> generatePerfectMultipleData(int size, double[] coefficients) {
        return generateMultipleRegressionData(size, coefficients, 0.0);
    }

    public List<DataPoint> generateLinearData(int size, double a, double b, double noiseLevel) {
        List<DataPoint> dataPoints = new ArrayList<>(size);

        for (int i = 0; i < size; i++) {
            double x = i * 0.1;
            double y = a + b * x;
            double noise = random.nextGaussian() * y * noiseLevel;
            y += noise;
            dataPoints.add(new DataPoint(new double[]{x}, y));
        }

        return dataPoints;
    }

    public List<DataPoint> generatePerfectLinearData(int size, double a, double b) {
        return generateLinearData(size, a, b, 0.0);
    }
}
