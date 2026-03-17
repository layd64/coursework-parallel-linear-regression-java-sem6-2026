package ua.coursework.regression.parallel;

import ua.coursework.regression.model.DataPoint;
import ua.coursework.regression.model.RegressionResult;
import ua.coursework.regression.sequential.SequentialRegression;

import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.DoubleAdder;


public class ParallelRegression {

    private final int numberOfThreads;
    private final int taskMultiplier;


    public ParallelRegression(int numberOfThreads, int taskMultiplier) {
        if (numberOfThreads < 1) {
            throw new IllegalArgumentException("Number of threads must be >= 1");
        }
        if (taskMultiplier < 1) {
            throw new IllegalArgumentException("Task multiplier must be >= 1");
        }
        this.numberOfThreads = numberOfThreads;
        this.taskMultiplier = taskMultiplier;
    }

    public ParallelRegression(int numberOfThreads) {
        this(numberOfThreads, 10);
    }

    public ParallelRegression() {
        this(Runtime.getRuntime().availableProcessors(), 10);
    }


    public RegressionResult calculate(List<DataPoint> dataPoints) {
        if (dataPoints == null || dataPoints.isEmpty()) {
            throw new IllegalArgumentException("Data points list cannot be empty");
        }

        int m = dataPoints.size();
        int n = dataPoints.get(0).getNumberOfVariables();
        int p = n + 1;

        if (m <= p) {
            throw new IllegalArgumentException(
                    "Number of observations (m) must be strictly greater than the number of variables + 1 (m > n + 1) to calculate variance and t-statistics.");
        }

        long startTime = System.nanoTime();

        ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);

        try {

            DoubleAdder[] sumXAdders = new DoubleAdder[n];
            for (int j = 0; j < n; j++) {
                sumXAdders[j] = new DoubleAdder();
            }
            DoubleAdder sumYAdder = new DoubleAdder();

            int numTasks = numberOfThreads * taskMultiplier;
            if (numTasks > m) numTasks = m;
            int baseSize = m / numTasks;
            int remainder = m % numTasks;
            CountDownLatch meanLatch = new CountDownLatch(numTasks);

            int offset = 0;
            for (int i = 0; i < numTasks; i++) {
                int currentChunkSize = baseSize + (i < remainder ? 1 : 0);
                int startIndex = offset;
                int endIndex = offset + currentChunkSize;
                offset = endIndex;

                List<DataPoint> chunk = dataPoints.subList(startIndex, endIndex);
                final int numVars = n;
                executor.submit(() -> {
                    double[] localSumX = new double[numVars];
                    double localSumY = 0;

                    for (DataPoint point : chunk) {
                        for (int j = 0; j < numVars; j++) {
                            localSumX[j] += point.getX(j);
                        }
                        localSumY += point.getY();
                    }

                    for (int j = 0; j < numVars; j++) {
                        sumXAdders[j].add(localSumX[j]);
                    }
                    sumYAdder.add(localSumY);
                    meanLatch.countDown();
                });
            }

            meanLatch.await();

            double[] meansX = new double[n];
            for (int j = 0; j < n; j++) {
                meansX[j] = sumXAdders[j].sum() / m;
            }
            double meanY = sumYAdder.sum() / m;


            DoubleAdder[] varXAdders = new DoubleAdder[n];
            for (int j = 0; j < n; j++) {
                varXAdders[j] = new DoubleAdder();
            }
            DoubleAdder varYAdder = new DoubleAdder();

            CountDownLatch stdLatch = new CountDownLatch(numTasks);

            offset = 0;
            for (int i = 0; i < numTasks; i++) {
                int currentChunkSize = baseSize + (i < remainder ? 1 : 0);
                int startIndex = offset;
                int endIndex = offset + currentChunkSize;
                offset = endIndex;

                List<DataPoint> chunk = dataPoints.subList(startIndex, endIndex);
                final int numVars = n;
                final double[] meansCopy = meansX.clone();
                final double meanYCopy = meanY;
                executor.submit(() -> {
                    double[] localVarX = new double[numVars];
                    double localVarY = 0;

                    for (DataPoint point : chunk) {
                        for (int j = 0; j < numVars; j++) {
                            localVarX[j] += Math.pow(point.getX(j) - meansCopy[j], 2);
                        }
                        localVarY += Math.pow(point.getY() - meanYCopy, 2);
                    }

                    for (int j = 0; j < numVars; j++) {
                        varXAdders[j].add(localVarX[j]);
                    }
                    varYAdder.add(localVarY);
                    stdLatch.countDown();
                });
            }

            stdLatch.await();

            double[] stdDevsX = new double[n];
            for (int j = 0; j < n; j++) {
                stdDevsX[j] = Math.sqrt(varXAdders[j].sum() / (m - 1));
                if (stdDevsX[j] == 0)
                    stdDevsX[j] = 1;
            }
            double stdDevY = Math.sqrt(varYAdder.sum() / (m - 1));
            if (stdDevY == 0)
                stdDevY = 1;


            DoubleAdder[][] xtxAdders = new DoubleAdder[p][p];
            DoubleAdder[] xtyAdders = new DoubleAdder[p];
            for (int j = 0; j < p; j++) {
                xtyAdders[j] = new DoubleAdder();
                for (int k = 0; k < p; k++) {
                    xtxAdders[j][k] = new DoubleAdder();
                }
            }
            CountDownLatch matrixLatch = new CountDownLatch(numTasks);

            offset = 0;
            for (int i = 0; i < numTasks; i++) {
                int currentChunkSize = baseSize + (i < remainder ? 1 : 0);
                int startIndex = offset;
                int endIndex = offset + currentChunkSize;
                offset = endIndex;

                List<DataPoint> chunk = dataPoints.subList(startIndex, endIndex);
                executor.submit(new MatrixAccumulationTask(
                        chunk, n, p, meansX, stdDevsX, meanY, stdDevY,
                        xtxAdders, xtyAdders, matrixLatch));
            }

            matrixLatch.await();


            double[][] xtx = new double[p][p];
            double[] xty = new double[p];
            for (int j = 0; j < p; j++) {
                xty[j] = xtyAdders[j].sum();
                for (int k = 0; k < p; k++) {
                    xtx[j][k] = xtxAdders[j][k].sum();
                }
            }


            double[] normCoefficients = solveLinearSystemParallel(xtx, xty, p, executor);


            double[] coefficients = new double[p];
            final double finalStdDevY = stdDevY;
            final double finalMeanY = meanY;

            CountDownLatch denormLatch = new CountDownLatch(n);
            for (int j = 1; j < p; j++) {
                final int jj = j;
                executor.submit(() -> {
                    coefficients[jj] = normCoefficients[jj] * finalStdDevY / stdDevsX[jj - 1];
                    denormLatch.countDown();
                });
            }
            denormLatch.await();

            DoubleAdder interceptAdder = new DoubleAdder();
            interceptAdder.add(finalMeanY + finalStdDevY * normCoefficients[0]);
            CountDownLatch interceptLatch = new CountDownLatch(n);
            for (int j = 1; j < p; j++) {
                final int jj = j;
                executor.submit(() -> {
                    interceptAdder.add(-coefficients[jj] * meansX[jj - 1]);
                    interceptLatch.countDown();
                });
            }
            interceptLatch.await();
            coefficients[0] = interceptAdder.sum();

            int finalNumTasks = numTasks;

            long endTime = System.nanoTime();
            double computationTime = (endTime - startTime) / 1_000_000.0;

            RegressionResult result = new RegressionResult(coefficients, computationTime, m, n);
            result.setNormalizationParams(meansX, stdDevsX, meanY, stdDevY);


            double rSquared = calculateRSquaredParallel(dataPoints, result, meanY, executor,
                    finalNumTasks);
            result.setRSquared(rSquared);

            double adjustedR2 = 1 - (1 - rSquared) * (m - 1.0) / (m - p - 1);
            result.setAdjustedRSquared(adjustedR2);


            calculateSignificanceParallel(dataPoints, result, xtx, executor,
                    finalNumTasks);

            return result;

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel computations were interrupted", e);
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }


    private double calculateRSquaredParallel(List<DataPoint> dataPoints,
            RegressionResult result,
            double meanY,
            ExecutorService executor,
            int numTasks)
            throws InterruptedException {

        int m = dataPoints.size();
        int baseSize = m / numTasks;
        int remainder = m % numTasks;

        DoubleAdder ssTotalAdder = new DoubleAdder();
        DoubleAdder ssResidualAdder = new DoubleAdder();
        CountDownLatch r2Latch = new CountDownLatch(numTasks);

        int offset = 0;
        for (int i = 0; i < numTasks; i++) {
            int currentChunkSize = baseSize + (i < remainder ? 1 : 0);
            int startIndex = offset;
            int endIndex = offset + currentChunkSize;
            offset = endIndex;

            List<DataPoint> chunk = dataPoints.subList(startIndex, endIndex);
            executor.submit(new RSquaredTask(chunk, result, meanY,
                    ssTotalAdder, ssResidualAdder, r2Latch));
        }
        r2Latch.await();

        double ssTotal = ssTotalAdder.sum();
        double ssResidual = ssResidualAdder.sum();

        if (ssTotal == 0)
            return 1.0;
        return 1 - (ssResidual / ssTotal);
    }


    private void calculateSignificanceParallel(List<DataPoint> dataPoints,
            RegressionResult result,
            double[][] xtx,
            ExecutorService executor,
            int numTasks)
            throws InterruptedException {

        int m = dataPoints.size();
        int p = result.getNumberOfVariables() + 1;
        int baseSize = m / numTasks;
        int remainder = m % numTasks;

        DoubleAdder ssResAdder = new DoubleAdder();
        CountDownLatch sigLatch = new CountDownLatch(numTasks);

        int offset = 0;
        for (int i = 0; i < numTasks; i++) {
            int currentChunkSize = baseSize + (i < remainder ? 1 : 0);
            int startIndex = offset;
            int endIndex = offset + currentChunkSize;
            offset = endIndex;

            List<DataPoint> chunk = dataPoints.subList(startIndex, endIndex);
            executor.submit(() -> {
                double localSsRes = 0;
                for (DataPoint point : chunk) {
                    double yPred = result.predict(point);
                    double residual = point.getY() - yPred;
                    localSsRes += residual * residual;
                }
                ssResAdder.add(localSsRes);
                sigLatch.countDown();
            });
        }
        sigLatch.await();

        double sigmaSquared = ssResAdder.sum() / (m - p);


        double[][] xtxInverse = SequentialRegression.invertMatrix(xtx, p);

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
                pValues[i] = SequentialRegression.calculatePValue(Math.abs(tStatistics[i]), df);
            }
        }

        result.setStandardErrors(standardErrors);
        result.setTStatistics(tStatistics);
        result.setPValues(pValues);
    }


    private static double[] solveLinearSystemParallel(double[][] A, double[] b, int n,
            ExecutorService executor) throws InterruptedException {

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

            int remainingRows = n - k - 1;
            if (remainingRows > 0) {
                CountDownLatch latch = new CountDownLatch(remainingRows);
                final int kk = k;
                for (int i = k + 1; i < n; i++) {
                    final int ii = i;
                    executor.submit(() -> {
                        double factor = a[ii][kk] / a[kk][kk];
                        for (int j = kk; j < n; j++) {
                            a[ii][j] -= factor * a[kk][j];
                        }
                        rhs[ii] -= factor * rhs[kk];
                        latch.countDown();
                    });
                }
                latch.await();
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

    public int getNumberOfThreads() {
        return numberOfThreads;
    }

    public int getTaskMultiplier() {
        return taskMultiplier;
    }




    private static class MatrixAccumulationTask implements Runnable {
        private final List<DataPoint> dataPoints;
        private final int n;
        private final int p;
        private final double[] meansX;
        private final double[] stdDevsX;
        private final double meanY;
        private final double stdDevY;
        private final DoubleAdder[][] xtxAdders;
        private final DoubleAdder[] xtyAdders;
        private final CountDownLatch latch;

        MatrixAccumulationTask(List<DataPoint> dataPoints, int n, int p,
                double[] meansX, double[] stdDevsX,
                double meanY, double stdDevY,
                DoubleAdder[][] xtxAdders, DoubleAdder[] xtyAdders,
                CountDownLatch latch) {
            this.dataPoints = dataPoints;
            this.n = n;
            this.p = p;
            this.meansX = meansX;
            this.stdDevsX = stdDevsX;
            this.meanY = meanY;
            this.stdDevY = stdDevY;
            this.xtxAdders = xtxAdders;
            this.xtyAdders = xtyAdders;
            this.latch = latch;
        }

        @Override
        public void run() {

            double[][] localXtX = new double[p][p];
            double[] localXtY = new double[p];

            double[] normX = new double[n];
            for (DataPoint point : dataPoints) {

                for (int j = 0; j < n; j++) {
                    normX[j] = (point.getX(j) - meansX[j]) / stdDevsX[j];
                }
                double normYVal = (point.getY() - meanY) / stdDevY;


                localXtX[0][0] += 1;
                localXtY[0] += normYVal;

                for (int j = 0; j < n; j++) {
                    localXtX[0][j + 1] += normX[j];
                    localXtX[j + 1][0] += normX[j];
                    localXtY[j + 1] += normX[j] * normYVal;

                    for (int k = 0; k < n; k++) {
                        localXtX[j + 1][k + 1] += normX[j] * normX[k];
                    }
                }
            }


            for (int j = 0; j < p; j++) {
                xtyAdders[j].add(localXtY[j]);
                for (int k = 0; k < p; k++) {
                    xtxAdders[j][k].add(localXtX[j][k]);
                }
            }

            latch.countDown();
        }
    }


    private static class RSquaredTask implements Runnable {
        private final List<DataPoint> dataPoints;
        private final RegressionResult result;
        private final double meanY;
        private final DoubleAdder atomicSsTotal;
        private final DoubleAdder atomicSsResidual;
        private final CountDownLatch latch;

        RSquaredTask(List<DataPoint> dataPoints, RegressionResult result,
                double meanY,
                DoubleAdder atomicSsTotal, DoubleAdder atomicSsResidual,
                CountDownLatch latch) {
            this.dataPoints = dataPoints;
            this.result = result;
            this.meanY = meanY;
            this.atomicSsTotal = atomicSsTotal;
            this.atomicSsResidual = atomicSsResidual;
            this.latch = latch;
        }

        @Override
        public void run() {

            double localSsTotal = 0;
            double localSsResidual = 0;

            for (DataPoint point : dataPoints) {
                double y = point.getY();
                double yPredicted = result.predict(point);

                localSsTotal += Math.pow(y - meanY, 2);
                localSsResidual += Math.pow(y - yPredicted, 2);
            }


            atomicSsTotal.add(localSsTotal);
            atomicSsResidual.add(localSsResidual);

            latch.countDown();
        }
    }
}
