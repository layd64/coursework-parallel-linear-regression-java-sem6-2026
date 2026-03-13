package ua.coursework.regression;

import ua.coursework.regression.benchmark.PerformanceBenchmark;
import java.util.List;


public class AutomatedExperiments {

    public static void runSuperTest() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║        ALL-IN-ONE SUPER TEST (FULL COURSEWORK REPORT)      ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println();

        PerformanceBenchmark benchmark = new PerformanceBenchmark();
        benchmark.warmUpJVM();

        System.out.println("Generating a single massive dataset (10,000,000 rows, 3 variables) ...");
        ua.coursework.regression.benchmark.DataGenerator generator = new ua.coursework.regression.benchmark.DataGenerator();
        double[] trueCoefficients = { 5.0, 2.0, -1.5, 4.0 };
        int maxDataSize = 10_000_000;
        List<ua.coursework.regression.model.DataPoint> massiveData = generator.generateMultipleRegressionData(
                maxDataSize, trueCoefficients, 1.0);
        System.out.println("Data generated successfully! Mem size: ~" + (maxDataSize * 32 / 1024 / 1024) + " MB");
        System.out.println();

        int optimalThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("Available processors: " + optimalThreads);
        System.out.println();


        System.out.println("=== EXPERIMENT 1: Mathematical Correctness ===");
        System.out.println();
        List<ua.coursework.regression.model.DataPoint> correctData = massiveData.subList(0, 50_000);

        ua.coursework.regression.sequential.SequentialRegression seq = new ua.coursework.regression.sequential.SequentialRegression();
        ua.coursework.regression.parallel.ParallelRegression par = new ua.coursework.regression.parallel.ParallelRegression(
                optimalThreads);

        ua.coursework.regression.model.RegressionResult seqRes = seq.calculate(correctData);
        ua.coursework.regression.model.RegressionResult parRes = par.calculate(correctData);

        System.out.println("Sub-dataset size: 50,000 (extracted from main 10M array)");
        System.out.println("Sequential Eq: " + seqRes.getEquation());
        System.out.println("Parallel Eq:   " + parRes.getEquation());
        System.out.println("Sequential R2: " + String.format("%.6f", seqRes.getRSquared()));
        System.out.println("Parallel R2:   " + String.format("%.6f", parRes.getRSquared()));

        double[] sCoeffs = seqRes.getCoefficients();
        double[] pCoeffs = parRes.getCoefficients();
        boolean isMathCorrect = true;
        for (int i = 0; i < sCoeffs.length; i++) {
            double diff = Math.abs(sCoeffs[i] - pCoeffs[i]);
            System.out.printf("Difference b%d: %.8f%n", i, diff);
            if (diff > 1e-5)
                isMathCorrect = false;
        }
        System.out.println(isMathCorrect ? "[OK] Math is 100% correct!" : "[!!] Warning: Math mismatch!");
        System.out.println();
        System.out.println("─────────────────────────────────────────────────────────────");
        System.out.println();


        System.out.println("=== EXPERIMENT 2: Data Scalability ===");
        System.out.println();

        int[] dataSizes = {
                1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000
        };

        java.util.List<PerformanceBenchmark.BenchmarkResult> scalabilityResults = new java.util.ArrayList<>();
        for (int size : dataSizes) {
            List<ua.coursework.regression.model.DataPoint> subData = massiveData.subList(0, size);
            PerformanceBenchmark.BenchmarkResult res = benchmark.runBenchmarkWithData(subData, optimalThreads, 3);
            scalabilityResults.add(res);
            System.out.println(res);
        }

        System.out.println();
        System.out.println("─────────────────────────────────────────────────────────────");
        System.out.println();


        System.out.println("=== EXPERIMENT 3: Thread Scaling ===");
        System.out.println();

        int[] threadCounts = { 1, 2, 4, 6, 8, 12, 16 };

        List<ua.coursework.regression.model.DataPoint> threadData = massiveData.subList(0, 5_000_000);
        System.out.println("Testing on fixed subset: 5,000,000 rows");
        System.out.println();

        java.util.List<PerformanceBenchmark.BenchmarkResult> threadScalingResults = new java.util.ArrayList<>();
        for (int threads : threadCounts) {
            PerformanceBenchmark.BenchmarkResult res = benchmark.runBenchmarkWithData(threadData, threads, 3);
            threadScalingResults.add(res);
            System.out.println(res);
        }

        System.out.println();
        System.out.println("─────────────────────────────────────────────────────────────");
        System.out.println();


        System.out.println("=== EXPERIMENT 4: Variable Scaling ===");
        System.out.println();

        int[] variableCounts = { 3, 5, 10, 20, 50 };
        int fixedScalingDataSize = 1_000_000;
        System.out.printf("Testing on generated dataset: %,d rows%n", fixedScalingDataSize);
        System.out.println();

        java.util.List<PerformanceBenchmark.BenchmarkResult> variableScalingResults = new java.util.ArrayList<>();
        for (int vars : variableCounts) {
            PerformanceBenchmark.BenchmarkResult res = benchmark.runBenchmark(fixedScalingDataSize, optimalThreads,
                    vars);
            variableScalingResults.add(res);
            System.out.println(res);
        }

        System.out.println();
        System.out.println("─────────────────────────────────────────────────────────────");
        System.out.println();


        System.out.println("=== EXPERIMENT 5: Detailed Summary ===");
        System.out.println();

        analyzeSpeedup(scalabilityResults);
        System.out.println();
        analyzeEfficiency(threadScalingResults);
        System.out.println();
        analyzeVariableScaling(variableScalingResults);

        System.out.println();
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║                 SUPER TEST SUCCESSFULLY COMPLETED          ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
    }

    private static void analyzeSpeedup(List<PerformanceBenchmark.BenchmarkResult> results) {
        System.out.println("Speedup Analysis (Multiple regression, 3 variables):");
        System.out.println();
        System.out.printf("%-15s | %-15s | %-15s | %-10s%n",
                "Data Size", "Sequential", "Parallel", "Speedup");
        System.out.println("─".repeat(70));

        double totalSpeedup = 0;
        int successfulSpeedups = 0;

        for (PerformanceBenchmark.BenchmarkResult result : results) {
            System.out.printf("%-15s | %-15.3f | %-15.3f | %.2fx%n",
                    String.format("%,d", result.dataSize),
                    result.sequentialTimeMs,
                    result.parallelTimeMs,
                    result.speedup);

            if (result.speedup > 1.0) {
                totalSpeedup += result.speedup;
                successfulSpeedups++;
            }
        }

        System.out.println();
        if (successfulSpeedups > 0) {
            double avgSpeedup = totalSpeedup / successfulSpeedups;
            System.out.printf("Average speedup (where > 1): %.2fx%n", avgSpeedup);
            System.out.printf("Number of tests with speedup > 1: %d of %d%n",
                    successfulSpeedups, results.size());
        }
    }

    private static void analyzeEfficiency(List<PerformanceBenchmark.BenchmarkResult> results) {
        System.out.println("Parallelization Efficiency Analysis:");
        System.out.println();
        System.out.printf("%-10s | %-15s | %-15s | %-12s | %-12s%n",
                "Threads", "Sequential", "Parallel", "Speedup", "Efficiency");
        System.out.println("─".repeat(75));


        double t1Parallel = results.stream()
                .filter(r -> r.numberOfThreads == 1)
                .findFirst()
                .map(r -> r.parallelTimeMs)
                .orElse(results.get(0).parallelTimeMs);

        for (PerformanceBenchmark.BenchmarkResult result : results) {

            double efficiency = t1Parallel / (result.numberOfThreads * result.parallelTimeMs);
            System.out.printf("%-10d | %-15.3f | %-15.3f | %-12.2f | %-12.2f%%%n",
                    result.numberOfThreads,
                    result.sequentialTimeMs,
                    result.parallelTimeMs,
                    result.speedup,
                    efficiency * 100);
        }

        System.out.println();
        System.out.println("Efficiency = T(1 thread parallel) / (N * T(N threads parallel))");
        System.out.println("Ideal efficiency = 100% (linear speedup)");
    }

    private static void analyzeVariableScaling(List<PerformanceBenchmark.BenchmarkResult> results) {
        System.out.println("Variable Scaling Analysis (Fixed data size, optimal threads):");
        System.out.println();
        System.out.printf("%-10s | %-15s | %-15s | %-10s%n",
                "Variables", "Sequential", "Parallel", "Speedup");
        System.out.println("─".repeat(60));

        for (PerformanceBenchmark.BenchmarkResult result : results) {
            System.out.printf("%-10d | %-15.3f | %-15.3f | %.2fx%n",
                    result.numberOfVariables,
                    result.sequentialTimeMs,
                    result.parallelTimeMs,
                    result.speedup);
        }

        System.out.println();
        System.out.println("Note: As the number of variables (n) increases, the algorithmic complexity of");
        System.out.println("building the X^T * X matrix grows as O(m * n^2). The task shifts from being");
        System.out.println("memory-bound to compute-bound, allowing the CPU cores to be fully utilized.");
        System.out.println("Consequently, the parallel speedup significantly improves.");
    }
}
