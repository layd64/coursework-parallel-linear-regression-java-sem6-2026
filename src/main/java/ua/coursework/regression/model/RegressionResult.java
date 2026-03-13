package ua.coursework.regression.model;

import java.util.Arrays;


public class RegressionResult {
    private final double[] coefficients;
    private final double computationTimeMs;
    private final int dataSize;
    private final int numberOfVariables;

    private double rSquared;
    private double adjustedRSquared;
    private double[] standardErrors;
    private double[] tStatistics;
    private double[] pValues;

    private double[] meansX;
    private double[] stdDevsX;
    private double meanY;
    private double stdDevY;
    private boolean normalized;


    public RegressionResult(double[] coefficients, double computationTimeMs,
            int dataSize, int numberOfVariables) {
        this.coefficients = Arrays.copyOf(coefficients, coefficients.length);
        this.computationTimeMs = computationTimeMs;
        this.dataSize = dataSize;
        this.numberOfVariables = numberOfVariables;
        this.normalized = false;
    }






    public double getCoefficient(int index) {
        return coefficients[index];
    }


    public double[] getCoefficients() {
        return Arrays.copyOf(coefficients, coefficients.length);
    }



    public double getComputationTimeMs() {
        return computationTimeMs;
    }

    public int getDataSize() {
        return dataSize;
    }

    public int getNumberOfVariables() {
        return numberOfVariables;
    }



    public double getRSquared() {
        return rSquared;
    }

    public void setRSquared(double rSquared) {
        this.rSquared = rSquared;
    }

    public double getAdjustedRSquared() {
        return adjustedRSquared;
    }

    public void setAdjustedRSquared(double adjustedRSquared) {
        this.adjustedRSquared = adjustedRSquared;
    }



    public double[] getStandardErrors() {
        return standardErrors != null ? Arrays.copyOf(standardErrors, standardErrors.length) : null;
    }

    public void setStandardErrors(double[] standardErrors) {
        this.standardErrors = Arrays.copyOf(standardErrors, standardErrors.length);
    }

    public double[] getTStatistics() {
        return tStatistics != null ? Arrays.copyOf(tStatistics, tStatistics.length) : null;
    }

    public void setTStatistics(double[] tStatistics) {
        this.tStatistics = Arrays.copyOf(tStatistics, tStatistics.length);
    }

    public double[] getPValues() {
        return pValues != null ? Arrays.copyOf(pValues, pValues.length) : null;
    }

    public void setPValues(double[] pValues) {
        this.pValues = Arrays.copyOf(pValues, pValues.length);
    }



    public void setNormalizationParams(double[] meansX, double[] stdDevsX,
            double meanY, double stdDevY) {
        this.meansX = Arrays.copyOf(meansX, meansX.length);
        this.stdDevsX = Arrays.copyOf(stdDevsX, stdDevsX.length);
        this.meanY = meanY;
        this.stdDevY = stdDevY;
        this.normalized = true;
    }

    public boolean isNormalized() {
        return normalized;
    }

    public double[] getMeansX() {
        return meansX != null ? Arrays.copyOf(meansX, meansX.length) : null;
    }

    public double[] getStdDevsX() {
        return stdDevsX != null ? Arrays.copyOf(stdDevsX, stdDevsX.length) : null;
    }

    public double getMeanY() {
        return meanY;
    }

    public double getStdDevY() {
        return stdDevY;
    }


    public double predict(DataPoint point) {
        if (point.getNumberOfVariables() != numberOfVariables) {
            throw new IllegalArgumentException(
                    "Очікується " + numberOfVariables + " змінних, отримано " + point.getNumberOfVariables());
        }
        double y = coefficients[0];
        for (int i = 0; i < numberOfVariables; i++) {
            y += coefficients[i + 1] * point.getX(i);
        }
        return y;
    }


    public double predict(double[] x) {
        if (x.length != numberOfVariables) {
            throw new IllegalArgumentException(
                    "Очікується " + numberOfVariables + " змінних, отримано " + x.length);
        }
        double y = coefficients[0];
        for (int i = 0; i < numberOfVariables; i++) {
            y += coefficients[i + 1] * x[i];
        }
        return y;
    }



    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Regression Eq: ").append(getEquation()).append("\n");
        sb.append(String.format("R2 = %.6f", rSquared));
        if (adjustedRSquared != 0) {
            sb.append(String.format(", R2(adj) = %.6f", adjustedRSquared));
        }
        sb.append("\n");
        sb.append(String.format("Computation Time: %.3f ms%n", computationTimeMs));
        sb.append(String.format("Points count: %d, Variables: %d", dataSize, numberOfVariables));


        if (tStatistics != null && pValues != null) {
            sb.append("\n\n--- Coefficients Significance ---\n");
            sb.append(String.format("%-10s %-12s %-15s %-12s %-10s%n",
                    "Coeff.", "Value", "Std. Error", "t-stat", "p-value"));
            for (int i = 0; i < coefficients.length; i++) {
                String name = (i == 0) ? "b0" : "b" + i;
                String significance = (pValues[i] < 0.05) ? " *" : "";
                sb.append(String.format("%-10s %-12.4f %-15.4f %-12.4f %-10.4f%s%n",
                        name, coefficients[i],
                        standardErrors != null ? standardErrors[i] : 0,
                        tStatistics[i], pValues[i], significance));
            }
            sb.append("* = significant at 0.05 level");
        }

        return sb.toString();
    }


    public String getEquation() {
        StringBuilder sb = new StringBuilder("y = ");
        sb.append(String.format("%.4f", coefficients[0]));
        for (int i = 1; i < coefficients.length; i++) {
            if (coefficients[i] >= 0) {
                sb.append(" + ");
            } else {
                sb.append(" - ");
            }
            sb.append(String.format("%.4f*x%d", Math.abs(coefficients[i]), i));
        }
        return sb.toString();
    }
}
