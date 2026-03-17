package ua.coursework.regression.benchmark;

import ua.coursework.regression.model.DataPoint;
import ua.coursework.regression.model.RegressionResult;
import ua.coursework.regression.parallel.ParallelRegression;
import ua.coursework.regression.sequential.SequentialRegression;

import java.util.Arrays;
import java.util.List;


public class PerformanceBenchmark {

    private static final int BENCHMARK_RUNS   = 20;
    private static final int LOCAL_WARMUP_RUNS = 5;


    private static final int BATCH_THRESHOLD = 100_000;

    private final SequentialRegression sequentialRegression;
    private final DataGenerator dataGenerator;

    public PerformanceBenchmark() {
        this.sequentialRegression = new SequentialRegression();
        this.dataGenerator = new DataGenerator();
    }


    public static class BenchmarkResult {
        public final int dataSize;
        public final int numberOfThreads;
        public final int numberOfVariables;
        public final int taskMultiplier;
        public final double sequentialTimeMs;
        public final double parallelTimeMs;
        public final double speedup;
        public final RegressionResult sequentialResult;
        public final RegressionResult parallelResult;

        public BenchmarkResult(int dataSize, int numberOfThreads, int numberOfVariables,
                double sequentialTimeMs, double parallelTimeMs,
                RegressionResult sequentialResult,
                RegressionResult parallelResult) {
            this(dataSize, numberOfThreads, numberOfVariables, 10,
                    sequentialTimeMs, parallelTimeMs, sequentialResult, parallelResult);
        }

        public BenchmarkResult(int dataSize, int numberOfThreads, int numberOfVariables,
                int taskMultiplier,
                double sequentialTimeMs, double parallelTimeMs,
                RegressionResult sequentialResult,
                RegressionResult parallelResult) {
            this.dataSize = dataSize;
            this.numberOfThreads = numberOfThreads;
            this.numberOfVariables = numberOfVariables;
            this.taskMultiplier = taskMultiplier;
            this.sequentialTimeMs = sequentialTimeMs;
            this.parallelTimeMs = parallelTimeMs;
            this.speedup = sequentialTimeMs / Math.max(0.001, parallelTimeMs);
            this.sequentialResult = sequentialResult;
            this.parallelResult = parallelResult;
        }

        @Override
        public String toString() {
            return String.format(
                    "Size: %,10d | Vars: %d | Threads: %2d | " +
                            "Seq.: %8.3f ms | Par.: %8.3f ms | " +
                            "Speedup: %5.2fx",
                    dataSize, numberOfVariables, numberOfThreads,
                    sequentialTimeMs, parallelTimeMs, speedup);
        }
    }


    public BenchmarkResult runBenchmark(int dataSize, int numberOfThreads) {
        return runBenchmark(dataSize, numberOfThreads, 3);
    }


    public BenchmarkResult runBenchmark(int dataSize, int numberOfThreads, int numberOfVariables) {
        double[] coefficients = new double[numberOfVariables + 1];
        coefficients[0] = 5.0;
        for (int i = 1; i <= numberOfVariables; i++) {
            coefficients[i] = 2.0 / i;
        }
        List<DataPoint> data = dataGenerator.generateMultipleRegressionData(
                dataSize, coefficients, 1.0);

        return runBenchmarkWithData(data, numberOfThreads, numberOfVariables);
    }


    public BenchmarkResult runBenchmarkWithData(List<DataPoint> data, int numberOfThreads, int numberOfVariables) {
        ParallelRegression parallelRegression = new ParallelRegression(numberOfThreads);


        int repetitions = batchRepetitions(data.size());


        for (int i = 0; i < LOCAL_WARMUP_RUNS; i++) {
            for (int r = 0; r < repetitions; r++) {
                sequentialRegression.calculate(data);
            }
            for (int r = 0; r < repetitions; r++) {
                parallelRegression.calculate(data);
            }
        }


        double[] seqTimes = new double[BENCHMARK_RUNS];
        double[] parTimes = new double[BENCHMARK_RUNS];
        RegressionResult lastSeqResult = null;
        RegressionResult lastParResult = null;

        for (int run = 0; run < BENCHMARK_RUNS; run++) {

            long seqStart = System.nanoTime();
            for (int r = 0; r < repetitions; r++) {
                lastSeqResult = sequentialRegression.calculate(data);
            }
            seqTimes[run] = (System.nanoTime() - seqStart) / 1_000_000.0 / repetitions;


            long parStart = System.nanoTime();
            for (int r = 0; r < repetitions; r++) {
                lastParResult = parallelRegression.calculate(data);
            }
            parTimes[run] = (System.nanoTime() - parStart) / 1_000_000.0 / repetitions;
        }

        Arrays.sort(seqTimes);
        Arrays.sort(parTimes);

        return new BenchmarkResult(
                data.size(),
                numberOfThreads,
                numberOfVariables,
                median(seqTimes),
                median(parTimes),
                lastSeqResult,
                lastParResult);
    }


    public BenchmarkResult runBenchmarkWithTaskMultiplier(List<DataPoint> data,
            int numberOfThreads, int numberOfVariables, int taskMultiplier) {
        ParallelRegression parallelRegression = new ParallelRegression(numberOfThreads, taskMultiplier);

        int repetitions = batchRepetitions(data.size());

        for (int i = 0; i < LOCAL_WARMUP_RUNS; i++) {
            for (int r = 0; r < repetitions; r++) {
                sequentialRegression.calculate(data);
            }
            for (int r = 0; r < repetitions; r++) {
                parallelRegression.calculate(data);
            }
        }

        double[] seqTimes = new double[BENCHMARK_RUNS];
        double[] parTimes = new double[BENCHMARK_RUNS];
        RegressionResult lastSeqResult = null;
        RegressionResult lastParResult = null;

        for (int run = 0; run < BENCHMARK_RUNS; run++) {
            long seqStart = System.nanoTime();
            for (int r = 0; r < repetitions; r++) {
                lastSeqResult = sequentialRegression.calculate(data);
            }
            seqTimes[run] = (System.nanoTime() - seqStart) / 1_000_000.0 / repetitions;

            long parStart = System.nanoTime();
            for (int r = 0; r < repetitions; r++) {
                lastParResult = parallelRegression.calculate(data);
            }
            parTimes[run] = (System.nanoTime() - parStart) / 1_000_000.0 / repetitions;
        }

        Arrays.sort(seqTimes);
        Arrays.sort(parTimes);

        return new BenchmarkResult(
                data.size(),
                numberOfThreads,
                numberOfVariables,
                taskMultiplier,
                median(seqTimes),
                median(parTimes),
                lastSeqResult,
                lastParResult);
    }


    private static int batchRepetitions(int dataSize) {
        if (dataSize >= BATCH_THRESHOLD) {
            return 1;
        }
        return Math.max(1, BATCH_THRESHOLD / dataSize);
    }


    private static double median(double[] sorted) {
        int n = sorted.length;
        if (n % 2 == 0) {
            return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        }
        return sorted[n / 2];
    }


    public void warmUpJVM() {
        System.out.println("JVM Warm-Up...");
        int warmUpSize = 500_000;
        int warmUpThreads = Runtime.getRuntime().availableProcessors();
        double[] coefficients = {5.0, 2.0, 1.0, 0.66};
        List<DataPoint> data = dataGenerator.generateMultipleRegressionData(warmUpSize, coefficients, 1.0);

        for (int i = 0; i < 5; i++) {
            sequentialRegression.calculate(data);
            new ParallelRegression(warmUpThreads).calculate(data);
        }
        System.out.println("Warm-Up completed.\n");
    }
}
