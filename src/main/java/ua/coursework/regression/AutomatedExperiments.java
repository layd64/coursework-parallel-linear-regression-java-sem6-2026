package ua.coursework.regression;

import ua.coursework.regression.benchmark.DataGenerator;
import ua.coursework.regression.benchmark.PerformanceBenchmark;
import ua.coursework.regression.model.DataPoint;
import ua.coursework.regression.model.RegressionResult;
import ua.coursework.regression.parallel.ParallelRegression;
import ua.coursework.regression.sequential.SequentialRegression;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AutomatedExperiments {

    public static void runSuperTest() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║        ALL-IN-ONE SUPER TEST (FULL COURSEWORK REPORT)      ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println();

        PerformanceBenchmark benchmark = new PerformanceBenchmark();
        benchmark.warmUpJVM();

        System.out.println("Generating a single massive dataset (10,000,000 rows, 3 variables) ...");
        DataGenerator generator = new DataGenerator();
        double[] trueCoefficients = { 5.0, 2.0, -1.5, 4.0 };
        int maxDataSize = 10_000_000;
        List<DataPoint> massiveData = generator.generateMultipleRegressionData(
                maxDataSize, trueCoefficients, 1.0);
        System.out.println("Data generated successfully! Mem size: ~" + (maxDataSize * 32 / 1024 / 1024) + " MB");
        System.out.println();

        int optimalThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("Available processors: " + optimalThreads);
        System.out.println();

        System.out.println("=== EXPERIMENT 1: Mathematical Correctness ===");
        System.out.println();
        List<DataPoint> correctData = massiveData.subList(0, 50_000);

        System.out.println("--- 1.1 Sequential Algorithm vs True Analytical Solution ---");
        System.out.println("Validating baseline sequential algorithm accuracy (3 variables)...");
        SequentialRegression seq = new SequentialRegression();
        RegressionResult seqRes = seq.calculate(correctData);

        double[] sCoeffs = seqRes.getCoefficients();
        boolean seqCorrect = true;
        double maxSeqErr = 0;

        System.out.printf("%-6s | %-12s | %-12s | %-12s%n", "Index", "True", "Sequential", "Absolute Error");
        System.out.println("─".repeat(55));
        for (int i = 0; i < sCoeffs.length; i++) {
            double err = Math.abs(trueCoefficients[i] - sCoeffs[i]);
            maxSeqErr = Math.max(maxSeqErr, err);
            System.out.printf("b%-5d | %12.4f | %12.4f | %12.4f%n", i, trueCoefficients[i], sCoeffs[i], err);
            if (err > 0.05) seqCorrect = false;
        }
        System.out.println(seqCorrect ? "[OK] Sequential baseline recovers true function accurately!" : "[!!] Warning: Sequential baseline is inaccurate!");
        System.out.println();

        System.out.println("--- 1.2 Parallel Algorithm vs Sequential Baseline ---");
        ParallelRegression par = new ParallelRegression(optimalThreads);
        RegressionResult parRes = par.calculate(correctData);

        System.out.println("Sub-dataset size: 50,000 (extracted from main 10M array)");
        System.out.println("Sequential Eq: " + seqRes.getEquation());
        System.out.println("Parallel Eq:   " + parRes.getEquation());
        System.out.println("Sequential R2: " + String.format("%.6f", seqRes.getRSquared()));
        System.out.println("Parallel R2:   " + String.format("%.6f", parRes.getRSquared()));

        double[] pCoeffs = parRes.getCoefficients();
        boolean isMathCorrect = true;
        for (int i = 0; i < sCoeffs.length; i++) {
            double diff = Math.abs(sCoeffs[i] - pCoeffs[i]);
            System.out.printf("Difference b%d: %.8f%n", i, diff);
            if (diff > 1e-5)
                isMathCorrect = false;
        }
        System.out.println(isMathCorrect ? "[OK] Parallel strictly matches sequential!" : "[!!] Warning: Math mismatch!");
        System.out.println();
        System.out.println("--- Correctness check with 100 coefficients (99 variables) ---");
        System.out.println();

        int largeVarCount = 99;
        int largeCoeffCount = largeVarCount + 1;
        double[] largeTrue = new double[largeCoeffCount];
        Random rng = new Random(123);
        for (int i = 0; i < largeCoeffCount; i++) {
            largeTrue[i] = Math.round((rng.nextDouble() * 20 - 10) * 100.0) / 100.0;
        }

        DataGenerator gen100 = new DataGenerator(777);
        List<DataPoint> data100 = gen100.generateMultipleRegressionData(
                500_000, largeTrue, 0.01);

        System.out.println("Sub-dataset size: 500,000 rows, " + largeVarCount + " variables");

        SequentialRegression seq100 = new SequentialRegression();
        ParallelRegression par100 = new ParallelRegression(optimalThreads);

        RegressionResult seqRes100 = seq100.calculate(data100);
        RegressionResult parRes100 = par100.calculate(data100);

        System.out.println("Sequential R2: " + String.format("%.6f", seqRes100.getRSquared()));
        System.out.println("Parallel R2:   " + String.format("%.6f", parRes100.getRSquared()));

        double[] sCoeffs100 = seqRes100.getCoefficients();
        double[] pCoeffs100 = parRes100.getCoefficients();

        boolean isMathCorrect100 = true;
        double maxDiffSeqPar = 0;
        double maxDiffTrue = 0;
        int mismatchCount = 0;

        for (int i = 0; i < sCoeffs100.length; i++) {
            double diffSeqPar = Math.abs(sCoeffs100[i] - pCoeffs100[i]);
            double diffTrue = Math.abs(sCoeffs100[i] - largeTrue[i]);
            maxDiffSeqPar = Math.max(maxDiffSeqPar, diffSeqPar);
            maxDiffTrue = Math.max(maxDiffTrue, diffTrue);
            if (diffSeqPar > 1e-5) {
                isMathCorrect100 = false;
                mismatchCount++;
            }
        }

        System.out.printf("Max |seq - par| difference: %.8f%n", maxDiffSeqPar);
        System.out.printf("Max |calculated - true| difference: %.4f%n", maxDiffTrue);
        System.out.println("Mismatched coefficients (seq vs par): " + mismatchCount + " / " + sCoeffs100.length);

        System.out.println("Last 10 coefficients comparison:");
        System.out.printf("%-6s | %-12s | %-12s | %-12s%n", "Index", "True", "Sequential", "Parallel");
        System.out.println("─".repeat(55));
        int startIdx = Math.max(0, sCoeffs100.length - 10);
        for (int i = startIdx; i < sCoeffs100.length; i++) {
            System.out.printf("b%-5d | %12.4f | %12.4f | %12.4f%n", i, largeTrue[i], sCoeffs100[i], pCoeffs100[i]);
        }

        System.out.println(isMathCorrect100
                ? "[OK] Math is 100% correct for 100 coefficients!"
                : "[!!] Warning: Math mismatch with 100 coefficients!");
        System.out.println();
        System.out.println("─────────────────────────────────────────────────────────────");
        System.out.println();

        System.out.println("=== EXPERIMENT 2: Data Scalability ===");
        System.out.println();

        int[] dataSizes = {
                1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000
        };

        List<PerformanceBenchmark.BenchmarkResult> scalabilityResults = new ArrayList<>();
        for (int size : dataSizes) {
            List<DataPoint> subData = massiveData.subList(0, size);
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

        List<DataPoint> threadData = massiveData.subList(0, 5_000_000);
        System.out.println("Testing on fixed subset: 5,000,000 rows");
        System.out.println();

        List<PerformanceBenchmark.BenchmarkResult> threadScalingResults = new ArrayList<>();
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

        List<PerformanceBenchmark.BenchmarkResult> variableScalingResults = new ArrayList<>();
        for (int vars : variableCounts) {
            PerformanceBenchmark.BenchmarkResult res = benchmark.runBenchmark(fixedScalingDataSize, optimalThreads,
                    vars);
            variableScalingResults.add(res);
            System.out.println(res);
        }

        System.out.println();
        System.out.println("─────────────────────────────────────────────────────────────");
        System.out.println();

        System.out.println("=== EXPERIMENT 5: Task Granularity (Subtask Count) ===");
        System.out.println();

        int[] taskMultipliers = { 1, 2, 5, 10, 20, 50, 100 };
        List<DataPoint> taskData = massiveData.subList(0, 5_000_000);
        System.out.printf("Fixed: %,d rows, %d threads, 3 variables%n", 5_000_000, optimalThreads);
        System.out.println("Varying task multiplier (subtasks = threads * multiplier)");
        System.out.println();

        List<PerformanceBenchmark.BenchmarkResult> taskGranularityResults = new ArrayList<>();
        for (int mult : taskMultipliers) {
            PerformanceBenchmark.BenchmarkResult res = benchmark.runBenchmarkWithTaskMultiplier(
                    taskData, optimalThreads, 3, mult);
            taskGranularityResults.add(res);
            System.out.printf("Multiplier: %3d | Subtasks: %5d | Seq.: %8.3f ms | Par.: %8.3f ms | Speedup: %5.2fx%n",
                    mult, optimalThreads * mult,
                    res.sequentialTimeMs, res.parallelTimeMs, res.speedup);
        }

        System.out.println();
        System.out.println("─────────────────────────────────────────────────────────────");
        System.out.println();

        System.out.println("=== EXPERIMENT 6: Detailed Summary ===");
        System.out.println();

        analyzeSpeedup(scalabilityResults);
        System.out.println();
        analyzeEfficiency(threadScalingResults);
        System.out.println();
        analyzeVariableScaling(variableScalingResults);
        System.out.println();
        analyzeTaskGranularity(taskGranularityResults);

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

        double baselineTime = results.stream()
                .filter(r -> r.numberOfThreads == 1)
                .findFirst()
                .map(r -> Math.min(r.sequentialTimeMs, r.parallelTimeMs))
                .orElse(results.get(0).sequentialTimeMs);

        for (PerformanceBenchmark.BenchmarkResult result : results) {
            double speedup = baselineTime / result.parallelTimeMs;
            double efficiency = speedup / result.numberOfThreads;

            System.out.printf("%-10d | %-15.3f | %-15.3f | %-12.2fx | %-12.4f%n",
                    result.numberOfThreads,
                    result.sequentialTimeMs,
                    result.parallelTimeMs,
                    speedup,
                    efficiency);
        }
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
    }

    private static void analyzeTaskGranularity(List<PerformanceBenchmark.BenchmarkResult> results) {
        System.out.println("Task Granularity Analysis (Fixed threads, data size, variables):");
        System.out.println();
        System.out.printf("%-12s | %-12s | %-15s | %-15s | %-10s%n",
                "Multiplier", "Subtasks", "Sequential", "Parallel", "Speedup");
        System.out.println("─".repeat(75));

        for (PerformanceBenchmark.BenchmarkResult result : results) {
            int subtasks = result.numberOfThreads * result.taskMultiplier;
            System.out.printf("%-12d | %-12d | %-15.3f | %-15.3f | %.2fx%n",
                    result.taskMultiplier,
                    subtasks,
                    result.sequentialTimeMs,
                    result.parallelTimeMs,
                    result.speedup);
        }
    }
}
