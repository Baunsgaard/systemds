/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.compress.lib;

import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

public class CLALibReorg {

	protected static final Log LOG = LogFactory.getLog(CLALibReorg.class.getName());

	public static MatrixBlock reorg(CompressedMatrixBlock cmb, ReorgOperator op, MatrixBlock ret, int startRow,
		int startColumn, int length) {
		// SwapIndex is transpose
		if(op.fn instanceof SwapIndex && cmb.getNumColumns() == 1) {
			MatrixBlock tmp = cmb.decompress(op.getNumThreads());
			long nz = tmp.setNonZeros(tmp.getNonZeros());
			tmp = new MatrixBlock(tmp.getNumColumns(), tmp.getNumRows(), tmp.getDenseBlockValues());
			tmp.setNonZeros(nz);
			return tmp;
		}
		else if(op.fn instanceof SwapIndex) {
			if(cmb.getCachedDecompressed() != null)
				return cmb.getCachedDecompressed().reorgOperations(op, ret, startRow, startColumn, length);

			return transpose(cmb, ret, op.getNumThreads());
		}
		else {
			// Allow transpose to be compressed output. In general we need to have a transposed flag on
			// the compressed matrix. https://issues.apache.org/jira/browse/SYSTEMDS-3025
			String message = op.getClass().getSimpleName() + " -- " + op.fn.getClass().getSimpleName();
			MatrixBlock tmp = cmb.getUncompressed(message, op.getNumThreads());
			return tmp.reorgOperations(op, ret, startRow, startColumn, length);
		}
	}

	private static MatrixBlock transpose(CompressedMatrixBlock cmb, MatrixBlock ret, int k) {

		final long nnz = cmb.getNonZeros();
		final int nRow = cmb.getNumRows();
		final int nCol = cmb.getNumColumns();
		final boolean sparseOut = MatrixBlock.evalSparseFormatInMemory(nRow, nCol, nnz);
		if(sparseOut)
			return transposeSparse(cmb, ret, k);
		else
			return transposeDense(cmb, ret, k, nRow, nCol, nnz);
	}

	private static MatrixBlock transposeSparse(CompressedMatrixBlock cmb, MatrixBlock ret, int k) {
		throw new NotImplementedException();
	}

	private static MatrixBlock transposeDense(CompressedMatrixBlock cmb, MatrixBlock ret, int k, int nRow, int nCol,
		long nnz) {
		if(ret == null)
			ret = new MatrixBlock(nCol, nRow, false, nnz);
		else
			ret.reset(nCol, nRow, false, nnz);

		ret.allocateDenseBlock();

		decompressToTransposedDense(ret, cmb.getColGroups(), nRow, 0, nRow);
		return ret;
	}

	private static void decompressToTransposedDense(MatrixBlock ret, List<AColGroup> groups, int rlen, int rl, int ru) {
		for(int i = 0; i < groups.size(); i++) {
			AColGroup g = groups.get(i);
			g.decompressToDenseBlockTransposed(ret.getDenseBlock(), rl, ru);
		}
	}
}