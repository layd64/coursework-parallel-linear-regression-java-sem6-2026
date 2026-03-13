package ua.coursework.regression;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import ua.coursework.regression.benchmark.DataGenerator;
import ua.coursework.regression.model.DataPoint;
import ua.coursework.regression.model.RegressionResult;
import ua.coursework.regression.parallel.ParallelRegression;
import ua.coursework.regression.sequential.SequentialRegression;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RegressionTest {

    private static final double EPSILON = 0.01;



    @Test
    @DisplayName("Sequential: ідеальні дані (y = 3 + 4*x), ручна генерація")
    void testSequentialWithPerfectData() {
        List<DataPoint> data = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            double x = i;
            double y = 3 + 4 * x;
            data.add(new DataPoint(new double[]{x}, y));
        }

        SequentialRegression regression = new SequentialRegression();
        RegressionResult result = regression.calculate(data);

        assertEquals(3.0, result.getCoefficients()[0], EPSILON, "b0 has to be 3.0");
        assertEquals(4.0, result.getCoefficients()[1], EPSILON, "b1 has to be 4.0");
    }

    @Test
    @DisplayName("Sequential: множинна регресія (y = 2 + 3*x1 - 1*x2 + 0.5*x3), ідеальні дані")
    void testMultipleRegressionPerfect() {
        DataGenerator generator = new DataGenerator();
        double[] trueCoeffs = { 2.0, 3.0, -1.0, 0.5 };
        List<DataPoint> data = generator.generatePerfectMultipleData(5000, trueCoeffs);

        SequentialRegression sequential = new SequentialRegression();
        RegressionResult result = sequential.calculate(data);

        double[] coefficients = result.getCoefficients();
        assertEquals(2.0, coefficients[0], EPSILON, "b0 has to be ~2.0");
        assertEquals(3.0, coefficients[1], EPSILON, "b1 has to be ~3.0");
        assertEquals(-1.0, coefficients[2], EPSILON, "b2 has to be ~-1.0");
        assertEquals(0.5, coefficients[3], EPSILON, "b3 has to be ~0.5");
    }



    @Test
    @DisplayName("Parallel: ідеальні дані (y = 5 + 2*x1), 4 потоки")
    void testParallelWithPerfectData() {
        DataGenerator generator = new DataGenerator();
        List<DataPoint> data = generator.generatePerfectLinearData(1000, 5.0, 2.0);

        ParallelRegression regression = new ParallelRegression(4);
        RegressionResult result = regression.calculate(data);

        assertEquals(5.0, result.getCoefficients()[0], EPSILON, "b0 has to be 5.0");
        assertEquals(2.0, result.getCoefficients()[1], EPSILON, "b1 has to be 2.0");
    }



    @Test
    @DisplayName("Sequential == Parallel: коефіцієнти на зашумлених даних (noise=0.5)")
    void testMultipleRegressionParallelVsSequential() {
        DataGenerator generator = new DataGenerator();
        double[] trueCoeffs = { 5.0, 2.0, -1.5, 4.0 };
        List<DataPoint> data = generator.generateMultipleRegressionData(10000, trueCoeffs, 0.5);

        SequentialRegression sequential = new SequentialRegression();
        ParallelRegression parallel = new ParallelRegression(8);

        RegressionResult seqResult = sequential.calculate(data);
        RegressionResult parResult = parallel.calculate(data);

        double[] seqCoeffs = seqResult.getCoefficients();
        double[] parCoeffs = parResult.getCoefficients();

        for (int i = 0; i < seqCoeffs.length; i++) {
            assertEquals(seqCoeffs[i], parCoeffs[i], EPSILON,
                    "Coefficient b" + i + " must match");
        }
    }

    @Test
    @DisplayName("Sequential == Parallel: R² однаковий на зашумлених даних")
    void testRSquaredSequentialVsParallel() {
        DataGenerator generator = new DataGenerator();
        double[] coeffs = { 7.0, 1.5, -2.0, 3.0 };
        List<DataPoint> data = generator.generateMultipleRegressionData(5000, coeffs, 1.0);

        SequentialRegression sequential = new SequentialRegression();
        ParallelRegression parallel = new ParallelRegression();

        RegressionResult seqResult = sequential.calculate(data);
        RegressionResult parResult = parallel.calculate(data);

        assertEquals(seqResult.getRSquared(), parResult.getRSquared(), EPSILON,
                "R2 must match between Sequential and Parallel");
    }

    @Test
    @DisplayName("Sequential == Parallel: різна кількість потоків (1, 2, 4, 8) дає однаковий результат")
    void testDifferentThreadCounts() {
        DataGenerator generator = new DataGenerator();
        double[] coeffs = { 4.0, 2.5, -0.3 };
        List<DataPoint> data = generator.generateMultipleRegressionData(10000, coeffs, 0.3);

        SequentialRegression sequential = new SequentialRegression();
        RegressionResult seqResult = sequential.calculate(data);

        for (int threads = 1; threads <= 8; threads *= 2) {
            ParallelRegression parallel = new ParallelRegression(threads);
            RegressionResult parResult = parallel.calculate(data);

            double[] seqCoeffs = seqResult.getCoefficients();
            double[] parCoeffs = parResult.getCoefficients();

            for (int i = 0; i < seqCoeffs.length; i++) {
                assertEquals(seqCoeffs[i], parCoeffs[i], EPSILON,
                        "b" + i + " must match for " + threads + " threads");
            }
        }
    }



    @Test
    @DisplayName("R2 близький до 1.0 для ідеальних даних")
    void testRSquaredPerfectData() {
        DataGenerator generator = new DataGenerator();
        double[] coeffs = { 2.0, 3.0, -1.0 };
        List<DataPoint> data = generator.generatePerfectMultipleData(500, coeffs);

        SequentialRegression regression = new SequentialRegression();
        RegressionResult result = regression.calculate(data);

        assertTrue(result.getRSquared() > 0.999,
                "R2 must be >= 0.999 for perfect data, got: " + result.getRSquared());
    }



    @Test
    @DisplayName("p-value < 0.05 для явних коефіцієнтів (велике значення, малий шум)")
    void testSignificantCoefficients() {
        DataGenerator generator = new DataGenerator();
        double[] coeffs = { 5.0, 10.0, -8.0 };
        List<DataPoint> data = generator.generateMultipleRegressionData(1000, coeffs, 0.1);

        SequentialRegression regression = new SequentialRegression();
        RegressionResult result = regression.calculate(data);

        double[] pValues = result.getPValues();
        assertNotNull(pValues, "p-values must be computed");

        for (int i = 1; i < pValues.length; i++) {
            assertTrue(pValues[i] < 0.05,
                    "Coefficient b" + i + " must be significant (p=" + pValues[i] + ")");
        }
    }

    @Test
    @DisplayName("t-статистики, стандартні помилки та p-значення обчислюються і не null")
    void testStatisticsNotNull() {
        DataGenerator generator = new DataGenerator();
        double[] coeffs = { 3.0, 2.0, -1.0 };
        List<DataPoint> data = generator.generateMultipleRegressionData(500, coeffs, 0.5);

        SequentialRegression regression = new SequentialRegression();
        RegressionResult result = regression.calculate(data);

        assertNotNull(result.getTStatistics(), "t-statistics must be computed");
        assertNotNull(result.getStandardErrors(), "Standard errors must be computed");
        assertNotNull(result.getPValues(), "p-values must be computed");
        assertEquals(coeffs.length, result.getTStatistics().length,
                "Number of t-statistics must equal number of coefficients");
    }



    @Test
    @DisplayName("Параметри нормалізації зберігаються у результаті")
    void testNormalizationParametersStored() {
        DataGenerator generator = new DataGenerator();
        double[] coeffs = { 1.0, 2.0, 3.0 };
        List<DataPoint> data = generator.generateMultipleRegressionData(1000, coeffs, 0.1);

        SequentialRegression regression = new SequentialRegression();
        RegressionResult result = regression.calculate(data);

        assertTrue(result.isNormalized(), "Result must be marked as normalized");
        assertNotNull(result.getMeansX(), "MeansX must be stored");
        assertNotNull(result.getStdDevsX(), "StdDevsX must be stored");
    }



    @Test
    @DisplayName("predict(1, 1) = 1 + 2*1 + 3*1 = 6.0 для y = 1 + 2*x1 + 3*x2")
    void testPredictMultiple() {
        DataGenerator generator = new DataGenerator();
        double[] coeffs = { 1.0, 2.0, 3.0 };
        List<DataPoint> data = generator.generatePerfectMultipleData(1000, coeffs);

        SequentialRegression regression = new SequentialRegression();
        RegressionResult result = regression.calculate(data);

        double predicted = result.predict(new double[]{ 1.0, 1.0 });
        assertEquals(6.0, predicted, 0.1, "predict(1, 1) must be ~6.0");
    }

    @Test
    @DisplayName("predict(0)=2, predict(1)=5, predict(10)=32 для y = 2 + 3*x")
    void testPredictSimple() {
        List<DataPoint> data = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            data.add(new DataPoint(new double[]{i}, 2 + 3.0 * i));
        }

        SequentialRegression regression = new SequentialRegression();
        RegressionResult result = regression.calculate(data);

        assertEquals(2.0, result.predict(new double[]{0}), 0.1, "predict(0) must be 2.0");
        assertEquals(5.0, result.predict(new double[]{1}), 0.1, "predict(1) must be 5.0");
        assertEquals(32.0, result.predict(new double[]{10}), 0.1, "predict(10) must be 32.0");
    }



    @Test
    @DisplayName("Sequential: IllegalArgumentException для порожнього списку")
    void testEmptyDataList() {
        List<DataPoint> data = new ArrayList<>();
        SequentialRegression regression = new SequentialRegression();

        assertThrows(IllegalArgumentException.class, () -> regression.calculate(data),
                "Must throw exception for empty list");
    }

    @Test
    @DisplayName("Sequential: IllegalArgumentException для null")
    void testNullDataList() {
        SequentialRegression regression = new SequentialRegression();

        assertThrows(IllegalArgumentException.class, () -> regression.calculate(null),
                "Must throw exception for null");
    }

    @Test
    @DisplayName("Parallel: IllegalArgumentException для null")
    void testParallelNullDataList() {
        ParallelRegression parallel = new ParallelRegression(4);

        assertThrows(IllegalArgumentException.class, () -> parallel.calculate(null),
                "Parallel must throw exception for null");
    }

    @Test
    @DisplayName("Parallel: IllegalArgumentException для порожнього списку")
    void testParallelEmptyDataList() {
        List<DataPoint> data = new ArrayList<>();
        ParallelRegression parallel = new ParallelRegression(4);

        assertThrows(IllegalArgumentException.class, () -> parallel.calculate(data),
                "Parallel must throw exception for empty list");
    }



    @Test
    @DisplayName("1 000 000 рядків: parallel == sequential за коефіцієнтами, R2 > 0.99")
    void testLargeDataCorrectness() {
        DataGenerator generator = new DataGenerator();
        double[] coeffs = { 5.0, 2.0, -1.0, 3.0 };
        List<DataPoint> data = generator.generateMultipleRegressionData(1_000_000, coeffs, 1.0);

        SequentialRegression sequential = new SequentialRegression();
        ParallelRegression parallel = new ParallelRegression();

        RegressionResult seqResult = sequential.calculate(data);
        RegressionResult parResult = parallel.calculate(data);

        assertTrue(seqResult.getRSquared() > 0.99,
                "Sequential R2 must be > 0.99, got: " + seqResult.getRSquared());
        assertEquals(seqResult.getRSquared(), parResult.getRSquared(), EPSILON,
                "R2 must match between algorithms on 1M rows");

        double[] seqCoeffs = seqResult.getCoefficients();
        double[] parCoeffs = parResult.getCoefficients();
        for (int i = 0; i < seqCoeffs.length; i++) {
            assertEquals(seqCoeffs[i], parCoeffs[i], EPSILON,
                    "Coefficient b" + i + " must match on 1M rows");
        }
    }
}
