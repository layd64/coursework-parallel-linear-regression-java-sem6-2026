package ua.coursework.regression.sequential;

import ua.coursework.regression.model.DataPoint;
import ua.coursework.regression.model.RegressionResult;

import java.util.List;


public class SequentialRegression {


    public RegressionResult calculate(List<DataPoint> dataPoints) {
        if (dataPoints == null || dataPoints.isEmpty()) {
            throw new IllegalArgumentException("Data points list cannot be empty");
        }

        int m = dataPoints.size();
        int n = dataPoints.get(0).getNumberOfVariables();
        int p = n + 1;

        if (m <= p) {
            throw new IllegalArgumentException("Number of observations (m) must be strictly greater than the number of variables + 1 (m > n + 1) to calculate variance and t-statistics.");
        }

        long startTime = System.nanoTime();


        double[] meansX = new double[n];
        double meanY = 0;

        for (DataPoint point : dataPoints) {
            for (int j = 0; j < n; j++) {
                meansX[j] += point.getX(j);
            }
            meanY += point.getY();
        }
        for (int j = 0; j < n; j++) {
            meansX[j] /= m;
        }
        meanY /= m;


        double[] stdDevsX = new double[n];
        double stdDevY = 0;

        for (DataPoint point : dataPoints) {
            for (int j = 0; j < n; j++) {
                stdDevsX[j] += Math.pow(point.getX(j) - meansX[j], 2);
            }
            stdDevY += Math.pow(point.getY() - meanY, 2);
        }

        stdDevY = Math.sqrt(stdDevY / (m - 1));
        if (stdDevY == 0)
            stdDevY = 1;
        for (int j = 0; j < n; j++) {
            stdDevsX[j] = Math.sqrt(stdDevsX[j] / (m - 1));
            if (stdDevsX[j] == 0)
                stdDevsX[j] = 1;
        }



        double[][] xtx = new double[p][p];
        double[] xty = new double[p];

        for (int i = 0; i < m; i++) {
            DataPoint point = dataPoints.get(i);
            

            double normY_i = (point.getY() - meanY) / stdDevY;
            double[] normX_i = new double[n];
            for (int j = 0; j < n; j++) {
                normX_i[j] = (point.getX(j) - meansX[j]) / stdDevsX[j];
            }


            xtx[0][0] += 1;
            xty[0] += normY_i;

            for (int j = 0; j < n; j++) {
                xtx[0][j + 1] += normX_i[j];
                xtx[j + 1][0] += normX_i[j];
                xty[j + 1] += normX_i[j] * normY_i;

                for (int k = 0; k < n; k++) {
                    xtx[j + 1][k + 1] += normX_i[j] * normX_i[k];
                }
            }
        }


        double[] normCoefficients = solveLinearSystem(xtx, xty, p);



        double[] coefficients = new double[p];
        coefficients[0] = meanY + stdDevY * normCoefficients[0];
        for (int j = 1; j < p; j++) {
            coefficients[j] = normCoefficients[j] * stdDevY / stdDevsX[j - 1];
            coefficients[0] -= coefficients[j] * meansX[j - 1];
        }

        long endTime = System.nanoTime();
        double computationTime = (endTime - startTime) / 1_000_000.0;

        RegressionResult result = new RegressionResult(coefficients, computationTime, m, n);


        result.setNormalizationParams(meansX, stdDevsX, meanY, stdDevY);


        double rSquared = calculateRSquared(dataPoints, result, meanY);
        result.setRSquared(rSquared);

        double adjustedR2 = 1 - (1 - rSquared) * (m - 1.0) / (m - p - 1);
        result.setAdjustedRSquared(adjustedR2);


        calculateSignificance(dataPoints, result, xtx);

        return result;
    }


    public double calculateRSquared(List<DataPoint> dataPoints, RegressionResult result, double meanY) {
        if (dataPoints == null || dataPoints.isEmpty()) {
            throw new IllegalArgumentException("Data points list cannot be empty");
        }


        double ssTotal = 0;
        double ssResidual = 0;

        for (DataPoint point : dataPoints) {
            double y = point.getY();
            double yPredicted = result.predict(point);

            ssTotal += Math.pow(y - meanY, 2);
            ssResidual += Math.pow(y - yPredicted, 2);
        }

        if (ssTotal == 0)
            return 1.0;
        return 1 - (ssResidual / ssTotal);
    }


    private void calculateSignificance(List<DataPoint> dataPoints,
            RegressionResult result, double[][] xtx) {
        int m = dataPoints.size();
        int p = result.getNumberOfVariables() + 1;


        double ssResidual = 0;
        for (DataPoint point : dataPoints) {
            double yPredicted = result.predict(point);
            double residual = point.getY() - yPredicted;
            ssResidual += residual * residual;
        }
        double sigmaSquared = ssResidual / (m - p);


        double[][] xtxInverse = invertMatrix(xtx, p);

        double[] standardErrors = new double[p];
        double[] tStatistics = new double[p];
        double[] pValues = new double[p];

        double[] coefficients = result.getCoefficients();

        for (int i = 0; i < p; i++) {

            standardErrors[i] = Math.sqrt(Math.abs(sigmaSquared * xtxInverse[i][i]));


            if (standardErrors[i] > 1e-10) {
                tStatistics[i] = coefficients[i] / standardErrors[i];
            } else {
                tStatistics[i] = Double.POSITIVE_INFINITY;
            }

            int df = m - p;
            if (Double.isInfinite(tStatistics[i])) {
                pValues[i] = 0.0;
            } else {
                pValues[i] = calculatePValue(Math.abs(tStatistics[i]), df);
            }
        }

        result.setStandardErrors(standardErrors);
        result.setTStatistics(tStatistics);
        result.setPValues(pValues);
    }




    public static double[] solveLinearSystem(double[][] A, double[] b, int n) {

        double[][] a = new double[n][n];
        double[] rhs = new double[n];
        for (int i = 0; i < n; i++) {
            a[i] = A[i].clone();
            rhs[i] = b[i];
        }


        for (int k = 0; k < n; k++) {

            int maxRow = k;
            double maxVal = Math.abs(a[k][k]);
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(a[i][k]) > maxVal) {
                    maxVal = Math.abs(a[i][k]);
                    maxRow = i;
                }
            }


            double[] tempRow = a[k];
            a[k] = a[maxRow];
            a[maxRow] = tempRow;
            double tempVal = rhs[k];
            rhs[k] = rhs[maxRow];
            rhs[maxRow] = tempVal;


            if (Math.abs(a[k][k]) < 1e-12) {
                throw new ArithmeticException(
                        "Matrix is singular or nearly singular at column " + k +
                        ". Check for multicollinearity or duplicate variables.");
            }

            for (int i = k + 1; i < n; i++) {
                double factor = a[i][k] / a[k][k];
                for (int j = k; j < n; j++) {
                    a[i][j] -= factor * a[k][j];
                }
                rhs[i] -= factor * rhs[k];
            }
        }


        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = rhs[i];
            for (int j = i + 1; j < n; j++) {
                x[i] -= a[i][j] * x[j];
            }
            if (Math.abs(a[i][i]) > 1e-12) {
                x[i] /= a[i][i];
            }
        }

        return x;
    }


    public static double[][] invertMatrix(double[][] matrix, int n) {

        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = matrix[i][j];
            }
            aug[i][n + i] = 1;
        }


        for (int k = 0; k < n; k++) {

            int maxRow = k;
            double maxVal = Math.abs(aug[k][k]);
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(aug[i][k]) > maxVal) {
                    maxVal = Math.abs(aug[i][k]);
                    maxRow = i;
                }
            }

            double[] temp = aug[k];
            aug[k] = aug[maxRow];
            aug[maxRow] = temp;

            double pivot = aug[k][k];
            if (Math.abs(pivot) < 1e-12)
                continue;


            for (int j = 0; j < 2 * n; j++) {
                aug[k][j] /= pivot;
            }


            for (int i = 0; i < n; i++) {
                if (i == k)
                    continue;
                double factor = aug[i][k];
                for (int j = 0; j < 2 * n; j++) {
                    aug[i][j] -= factor * aug[k][j];
                }
            }
        }


        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = aug[i][n + j];
            }
        }

        return inverse;
    }


    public static double calculatePValue(double t, int df) {
        if (df <= 0)
            return 1.0;
        if (Double.isInfinite(t))
            return 0.0;


        double z;
        if (df > 30) {
            z = t;
        } else {

            z = t * (1 - 1.0 / (4.0 * df));
        }


        double p = 2 * (1 - normalCDF(Math.abs(z)));
        return Math.max(0, Math.min(1, p));
    }


    private static double normalCDF(double x) {
        if (x < -8)
            return 0;
        if (x > 8)
            return 1;

        double b0 = 0.2316419;
        double b1 = 0.319381530;
        double b2 = -0.356563782;
        double b3 = 1.781477937;
        double b4 = -1.821255978;
        double b5 = 1.330274429;

        double t = 1.0 / (1.0 + b0 * Math.abs(x));
        double pdf = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);

        double cdf = 1 - pdf * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));

        return x >= 0 ? cdf : 1 - cdf;
    }
}
