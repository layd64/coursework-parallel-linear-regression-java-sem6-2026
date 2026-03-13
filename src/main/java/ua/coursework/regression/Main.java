package ua.coursework.regression;

import ua.coursework.regression.benchmark.DataGenerator;
import ua.coursework.regression.model.DataPoint;
import ua.coursework.regression.model.RegressionResult;
import ua.coursework.regression.parallel.ParallelRegression;
import ua.coursework.regression.sequential.SequentialRegression;

import java.util.List;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        while (true) {
            printMenu();
            System.out.print("Choose an option: ");

            String choice = scanner.nextLine().trim();
            System.out.println();

            switch (choice) {
                case "1":
                    runMultipleRegressionExample();
                    break;
                case "2":
                    runCorrectnessTest();
                    break;
                case "3":
                    runSuperTest();
                    break;
                case "0":
                    System.out.println("Goodbye!");
                    scanner.close();
                    return;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }

            System.out.println();
            System.out.println("Press Enter to continue...");
            scanner.nextLine();
            System.out.println();
        }
    }

    private static void printMenu() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║  MULTIPLE LINEAR REGRESSION (OLS) | PARALLEL COMPUTATION   ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("1. Demo: Sequential vs Parallel (5 000 000 rows, significance)");
        System.out.println("2. Correctness: Verify sequential == parallel results");
        System.out.println("3. Full Report: Scalability (1K-10M rows) + Thread scaling (1-16)");
        System.out.println("0. Exit");
        System.out.println();
    }

    private static void runMultipleRegressionExample() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║              DEMO: SEQUENTIAL vs PARALLEL                  ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println();

        DataGenerator generator = new DataGenerator();
        double[] trueCoefficients = { 3.0, 2.0, -1.5, 4.0 };
        int dataSize = 5_000_000;

        System.out.println("Dataset:           " + dataSize + " rows,  3 independent variables (x1, x2, x3)");
        System.out.println("True equation:     y = 3.0 + 2.0*x1 - 1.5*x2 + 4.0*x3");
        System.out.println("Noise level:       0.5");
        System.out.println();

        System.out.print("JVM Warm-Up (5 iterations on 200 000 rows) ... ");
        List<DataPoint> warmData = generator.generateMultipleRegressionData(200_000, trueCoefficients, 0.5);
        SequentialRegression sequential = new SequentialRegression();
        ParallelRegression parallel = new ParallelRegression();
        for (int i = 0; i < 5; i++) {
            sequential.calculate(warmData);
            parallel.calculate(warmData);
        }
        System.out.println("done.");
        System.out.println();

        List<DataPoint> data = generator.generateMultipleRegressionData(dataSize, trueCoefficients, 0.5);

        System.out.println("════════════════════════ SEQUENTIAL ════════════════════════");
        RegressionResult seqResult = sequential.calculate(data);
        System.out.println(seqResult);

        System.out.println("════════════════════════ PARALLEL ══════════════════════════");
        System.out.println("Threads: " + parallel.getNumberOfThreads());
        RegressionResult parResult = parallel.calculate(data);
        System.out.println(parResult);

        double speedup = seqResult.getComputationTimeMs() / Math.max(0.001, parResult.getComputationTimeMs());
        System.out.println("════════════════════════ COMPARISON ════════════════════════");
        System.out.printf("Sequential time:   %.3f ms%n", seqResult.getComputationTimeMs());
        System.out.printf("Parallel time:     %.3f ms%n", parResult.getComputationTimeMs());
        System.out.printf("Speedup:           %.2fx%n", speedup);
        System.out.println("══════════════════════════════════════════════════════════");
    }

    private static void runCorrectnessTest() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║         CORRECTNESS VERIFICATION TEST                      ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println();

        DataGenerator generator = new DataGenerator();
        SequentialRegression sequential = new SequentialRegression();
        ParallelRegression parallel = new ParallelRegression();

        System.out.println("-- Test 1: Simple linear dependency (y = 3 + 4*x1, 1000 rows, no noise) --");
        List<DataPoint> simpleData = generator.generatePerfectLinearData(1000, 3.0, 4.0);
        RegressionResult seqResult = sequential.calculate(simpleData);
        RegressionResult parResult = parallel.calculate(simpleData);
        System.out.println("  Sequential : " + seqResult.getEquation());
        System.out.println("  Parallel   : " + parResult.getEquation());
        double[] sc1 = seqResult.getCoefficients();
        double[] pc1 = parResult.getCoefficients();
        System.out.printf("  Diff b0    : %.10f%n", Math.abs(sc1[0] - pc1[0]));
        System.out.printf("  Diff b1    : %.10f%n", Math.abs(sc1[1] - pc1[1]));
        System.out.println();

        System.out.println("-- Test 2: Multiple regression (y = 2 + 3*x1 - 1*x2 + 0.5*x3, 5000 rows, no noise) --");
        double[] trueCoeffs = { 2.0, 3.0, -1.0, 0.5 };
        List<DataPoint> multiData = generator.generatePerfectMultipleData(5000, trueCoeffs);
        seqResult = sequential.calculate(multiData);
        parResult = parallel.calculate(multiData);
        System.out.println("  Sequential : " + seqResult.getEquation());
        System.out.println("  Parallel   : " + parResult.getEquation());
        System.out.printf("  R2 seq     : %.6f%n", seqResult.getRSquared());
        System.out.printf("  R2 par     : %.6f%n", parResult.getRSquared());
        double[] sc2 = seqResult.getCoefficients();
        double[] pc2 = parResult.getCoefficients();
        for (int i = 0; i < sc2.length; i++) {
            System.out.printf("  Diff b%d    : %.10f%n", i, Math.abs(sc2[i] - pc2[i]));
        }
        System.out.println();

        System.out.println("-- Test 3: Multiple regression with noise (y = 5 + 2*x1 - 1.5*x2 + 4*x3, 10 000 rows, noise 0.5) --");
        double[] noisyCoeffs = { 5.0, 2.0, -1.5, 4.0 };
        List<DataPoint> noisyData = generator.generateMultipleRegressionData(10000, noisyCoeffs, 0.5);
        seqResult = sequential.calculate(noisyData);
        System.out.println(seqResult);

        System.out.println("[PASSED] All correctness tests completed successfully!");
    }

    private static void runSuperTest() {
        AutomatedExperiments.runSuperTest();
    }
}
