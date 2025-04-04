# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

# Autogenerated By   : src/main/python/generator/generator.py
# Autogenerated From : scripts/builtin/bivar.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def bivar(X: Matrix,
          S1: Matrix,
          S2: Matrix,
          T1: Matrix,
          T2: Matrix,
          verbose: bool):
    """
     For a given pair of attribute sets, compute bivariate statistics between all attribute pairs.
     Given, index1 = {A_11, A_12, ... A_1m} and index2 = {A_21, A_22, ... A_2n}
     compute bivariate stats for m*n pairs (A_1i, A_2j), (1<= i <=m) and (1<= j <=n).
    
    
    
    :param X: Input matrix
    :param S1: First attribute set {A_11, A_12, ... A_1m}
    :param S2: Second attribute set {A_21, A_22, ... A_2n}
    :param T1: Kind for attributes in S1
        (kind=1 for scale, kind=2 for nominal, kind=3 for ordinal)
    :param verbose: Print bivar stats
    :return: basestats_scale_scale as output with bivar stats
    :return: basestats_nominal_scale as output with bivar stats
    :return: basestats_nominal_nominal as output with bivar stats
    :return: basestats_ordinal_ordinal as output with bivar stats
    """

    params_dict = {'X': X, 'S1': S1, 'S2': S2, 'T1': T1, 'T2': T2, 'verbose': verbose}
    
    vX_0 = Matrix(X.sds_context, '')
    vX_1 = Matrix(X.sds_context, '')
    vX_2 = Matrix(X.sds_context, '')
    vX_3 = Matrix(X.sds_context, '')
    output_nodes = [vX_0, vX_1, vX_2, vX_3, ]

    op = MultiReturn(X.sds_context, 'bivar', output_nodes, named_input_nodes=params_dict)

    vX_0._unnamed_input_nodes = [op]
    vX_1._unnamed_input_nodes = [op]
    vX_2._unnamed_input_nodes = [op]
    vX_3._unnamed_input_nodes = [op]

    return op
