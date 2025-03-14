package org.apache.sysds.performance.matrix;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.performance.TimingUtils.StatsType;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;

public class Primitives {
	protected static final Log LOG = LogFactory.getLog(Primitives.class.getName());

	public static void main(String[] args) {
		// double[] vals = TestUtils.(100, -10, 10, 1.0, 3215);

		int rows = 30000;
		int cols = 100;
		MatrixBlock mb = TestUtils.round(TestUtils.generateTestMatrixBlock(rows, cols, -10, 10, 0.3, 3215));

		mb.denseToSparse(true);
		mb.setSparseBlock(new SparseBlockCSR(mb.getSparseBlock()));
		mb.recomputeNonZeros();
		SparseBlockCSR sb = (SparseBlockCSR) mb.getSparseBlock();

		// print output:
		// MatrixBlock ret = new MatrixBlock();
		// ret.reset(cols, cols, false);
		// LibMatrixMult.matrixMultTransposeSelf(mb, ret, true, false);
		// LOG.error(ret);
		// LOG.error(Arrays.toString(tsmmCSR(sb.rowPointers(), sb.indexes(),
		// sb.values(), cols)));

		// LibMatrixMult.matrixMultTransposeSelf(mb, ret, true);
		// LOG.error(mb);
		// LOG.error(ret);
		// LOG.error(Arrays.toString(tsmm(mb.getDenseBlockValues(), rows, cols)));
		// LOG.error(mb);
		int rep = 1000;
		// System.out.println(//
		// TimingUtils.stats(//
		// TimingUtils.time(() -> mb.transposeSelfMatrixMultOperations(null,
		// MMTSJType.LEFT), rep),
		// StatsType.MEAN_STD_Q1));

		// System.out.println(//
		// TimingUtils.stats(//
		// TimingUtils.time(() -> LibMatrixMult.matrixMultTransposeSelf(mb, new
		// MatrixBlock(cols, cols, false), true,
		// false), rep),
		// StatsType.MEAN_STD_Q1));

		System.out.println(//
				TimingUtils.stats(//
						TimingUtils.time(() -> tsmmCSR(sb.rowPointers(), sb.indexes(), sb.values(), cols), rep),
						StatsType.MEAN_STD_Q1));

		// System.out.println(//
		// TimingUtils.stats(//
		// TimingUtils.time(() -> LibMatrixMult.matrixMultTransposeSelf(mb, ret, true,
		// false), rep),
		// StatsType.MEAN_STD_Q1));
		// System.out.println(//
		// TimingUtils.stats(//
		// TimingUtils.time(() -> tsmmCSR(sb.rowPointers(), sb.indexes(), sb.values(),
		// cols), rep),
		// StatsType.MEAN_STD_Q1));

	}

	public static double[] tsmm(double[] a, int nRow, int nCol) {
		double[] ret = new double[nCol * nCol]; // O(m^2)
		for (int i = 0; i < nCol; i++) {
			int offO = i * nCol;
			for (int j = 0; j < nRow; j++) {// O(n^2m)
				int offB = j * nCol;
				double aV = a[offB + i];
				offB = offB + i;
				for (int k = offO + i; k < offO + nCol; k++, offB++)
					ret[k] += aV * a[offB];
			}
		}
		copyUpperHalf(ret, nCol); // O(m^2)
		return ret;
	}

	public static double[] tsmmCSR(int[] rowOff, int[] cols, double[] a, int nCol) {
		final double[] ret = new double[nCol * nCol]; // O(m^2)
		extracted(rowOff, cols, a, nCol, ret);
		copyUpperHalf(ret, nCol);
		return ret;
	}

	private static void extracted(int[] rowOff, int[] cols, double[] a, int nCol, final double[] ret) {
		for (int i = 0; i < rowOff.length - 1; i++) { // O(A_nnz n)
			final int s = rowOff[i];
			final int e = rowOff[i + 1];
			for (int j = s; j < e; j++) {
				final double aV = a[j];
				final int col = cols[j] * nCol;
				LibMatrixMult.vectMultiplyAdd(aV, a, ret, cols, j, col, e - j);
				// for(int k = j; k < e; k++) {
				// ret[cols[k] + col] += aV * a[k];
				// }
			}
		}
	}

	private static void copyBottomHalf(double[] ret, int nCol) {
		for (int i = 0; i < nCol; i++) {
			for (int j = i + 1; j < nCol; j++) {
				ret[i * nCol + j] = ret[i + j * nCol];
			}
		}
	}

	private static void copyUpperHalf(double[] ret, int nCol) {
		for (int i = 0; i < nCol; i++) {
			for (int j = i + 1; j < nCol; j++) {
				ret[i + j * nCol] = ret[i * nCol + j];
			}
		}
	}

	public static double[] mmCSR(int[] rowOff, int[] cols, double[] a, double[] b, int rowA, int colB) {
		double[] ret = new double[rowA * colB]; // O(nk)
		for (int i = 0; i < rowOff.length - 1; i++) { // O(A_nnz k + n)
			int offO = i * colB;
			for (int j = rowOff[i]; j < rowOff[i + 1]; j++) {
				double aV = a[j];
				int offB = cols[j] * colB;
				for (int k = offO; k < offO + colB; k++, offB++)
					ret[k] += aV * b[offB];
			}
		}
		return ret;
	}

	public static double[] mmCOO(int[] rows, int[] cols, double[] a, double[] b, int rowA, int colB) {
		double[] ret = new double[rowA * colB]; // O(nk)
		for (int i = 0; i < a.length; i++) { // O(A_nnz k)
			double aV = a[i];
			int offO = rows[i] * colB;
			int offB = cols[i] * colB;
			for (int j = offO; j < offO + colB; j++, offB++)
				ret[j] += aV * b[offB];
		}
		return ret;
	}

	public static double[] mm(double[] a, double[] b, int rowA, int colB) {
		double[] ret = new double[rowA * colB]; // O(nk)
		int cd = a.length / rowA; // common dimension
		for (int i = 0; i < rowA; i++) { // O(nmk)
			int offO = i * colB;
			int offA = i * cd;
			for (int k = 0; k < cd; k++, offA++) {
				int offB = k * colB;
				double av = a[offA];
				for (int j = offO; j < offO + colB; j++, offB++)
					ret[j] += av * b[offB];
			}
		}
		return ret;
	}

	public static CSR multCSR(int[] rOff, int[] cols, double[] a, double[] b, int nCol) {
		double[] ret = new double[a.length]; // O(A_nnz)
		int nnz = 0;
		for (int i = 0; i < rOff.length - 1; i++) { // O(n + A_nnz)
			int s = rOff[i];
			int e = rOff[i + 1];
			int off = i * nCol; // precompute row offset
			for (int j = s; j < e; j++)
				ret[j] = a[j] * b[off + cols[j]];
		}
		if (nnz < a.length) // compact
			return compactCSR(rOff, cols, ret, nnz); // O(n + A_nnz)
		return new CSR(rOff, cols, ret);
	}

	public static COO multCOO(int[] rows, int[] cols, double[] a, double[] b, int nCol) {
		double[] ret = new double[a.length]; // O(A_nnz)
		int nnz = 0;
		for (int i = 0; i < a.length; i++) // O(A_nnz)
			nnz += (ret[i] = a[i] * b[rows[i] * nCol + cols[i]]) == 0 ? 0 : 1;
		if (nnz < a.length) // compact
			return compactCOO(rows, cols, ret, nnz); // O(A_nnz)
		return new COO(rows, cols, ret);
	}

	public static double[] mult(double[] a, double[] b) {
		double[] ret = new double[a.length]; // O(A)
		for (int i = 0; i < a.length; i++) // O(A)
			ret[i] = a[i] * b[i];
		return ret;
	}

	public static double[] rowMaxCSR(int[] rOff, int[] cols, double[] a, int nCol, int nRow) {
		double[] r = new double[nRow]; // O(n)
		for (int i = 0; i < rOff.length - 1; i++) { // O(n + A_nnz)
			int s = rOff[i];
			int e = rOff[i + 1];
			r[i] = e - s == nCol ? Double.MIN_VALUE : 0;
			for (int j = s; j < e; j++)
				r[i] = a[j] > r[i] ? a[j] : r[i];
		}
		return r;
	}

	public static double[] colMaxCOO(int[] cols, double[] a, int nCol, int nRow) {
		int[] counts = new int[nCol]; // O(m)
		for (int i = 0; i < a.length; i++) // nnz col count O(A_nnz)
			counts[cols[i]]++;
		double[] r = new double[nCol]; // O(m)
		for (int i = 0; i < nCol; i++) // initialize result O(m)
			r[i] = counts[i] == nRow ? Double.MIN_VALUE : 0;
		for (int i = 0; i < a.length; i++) // process O(A_nnz)
			r[cols[i]] = a[i] > r[cols[i]] ? a[i] : r[i];
		return r;
	}

	public static double max(double[] a, int nCol, int nRow) {
		double r = nCol * nRow > a.length ? 0 : a[0]; // O(1)
		for (int i = 0; i < a.length; i++) // O(A)
			r = a[i] > r ? a[i] : r;
		return r;
	}

	public static double max(double[] a) {
		double r = a[0]; // O(1)
		for (int i = 1; i < a.length; i++) // O(A)
			r = a[i] > r ? a[i] : r;
		return r;
	}

	public static double[] rowSumCSR(int[] rOff, double[] a, int nRow) {
		double[] r = new double[nRow]; // O(n)
		for (int i = 0; i < rOff.length - 1; i++) // O(n + A_nnz)
			for (int j = rOff[i]; i < rOff[i + 1]; j++)
				r[i] += a[j];
		return r;
	}

	public static double[] colSum(int[] cols, double[] a, int nCol) {
		double[] r = new double[nCol]; // O(m)
		for (int i = 0; i < a.length; i++) // O(A_nnz)
			r[cols[i]] += a[i];
		return r;
	}

	public static double[] colSum(double[] a, int nCol) {
		double[] r = new double[nCol]; // O(m)
		for (int i = 0; i < a.length; i++) // O(A)
			r[i % nCol] += a[i];
		return r;
	}

	public static double sum(double[] a) {
		double r = 0; // O(1)
		for (int i = 0; i < a.length; i++) // O(A)
			r += a[i];
		return r;
	}

	public static double[] mvCSR(int[] rowOff, int[] cols,
			double[] a, double[] b, int rowA) {
		double[] ret = new double[rowA]; // O(nk)
		for (int i = 0; i < rowOff.length - 1; i++) { // O(A_nnz + n)
			for (int j = rowOff[i]; j < rowOff[i + 1]; j++) {
				ret[i] += a[j] * b[cols[j]];
			}
		}
		return ret;
	}

	public static double[] mvCSRVI(int[] rowOff, int[] cols, 
        int[] m, double[] d, double[] b, int rowA){
    double [] ret = new double[rowA]; // O(nk)
    for(int i = 0; i < rowOff.length-1; i++){ // O(A_nnz + n)
        for(int j = rowOff[i]; j < rowOff[i+1];j++){
            ret[i] += d[m[j]] * b[cols[j]];
        }
    }
    return ret;
}

	public static class CSR {
		public CSR(int[] rOff, int[] cols, double[] vals) {

		}
	}

	public static CSR compactCSR(int[] rOff, int[] cols, double[] vals, int nnz) {
		return null;
	}

	public static class COO {
		public COO(int[] rows, int[] cols, double[] vals) {

		}
	}

	public static COO compactCOO(int[] rows, int[] cols, double[] vals, int nnz) {
		return null;
	}

}
