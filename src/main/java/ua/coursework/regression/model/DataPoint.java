package ua.coursework.regression.model;

import java.util.Arrays;

public class DataPoint {
    private final double[] x;
    private final double y;

    public DataPoint(double[] x, double y) {
        if (x == null || x.length == 0) {
            throw new IllegalArgumentException("Variables array cannot be empty");
        }
        this.x = Arrays.copyOf(x, x.length);
        this.y = y;
    }

    public double getX(int index) {
        return x[index];
    }

    public int getNumberOfVariables() {
        return x.length;
    }

    public double getY() {
        return y;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < x.length; i++) {
            sb.append(String.format("x%d=%.2f", i + 1, x[i]));
            if (i < x.length - 1)
                sb.append(", ");
        }
        sb.append(String.format(", y=%.2f)", y));
        return sb.toString();
    }
}
