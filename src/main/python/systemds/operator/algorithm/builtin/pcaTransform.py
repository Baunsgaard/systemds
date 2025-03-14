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
# Autogenerated From : scripts/builtin/pcaTransform.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def pcaTransform(X: Matrix,
                 Clusters: Matrix,
                 Centering: Matrix,
                 ScaleFactor: Matrix):
    """
     Principal Component Analysis (PCA) for dimensionality reduction prediction
     This method is used to transpose data, which the PCA model was not trained on. To validate how good
     The PCA is, and to apply in production. 
    
    
    
    :param X: Input feature matrix
    :param Clusters: The previously computed principal components
    :param Centering: The column means of the PCA model, subtracted to construct the PCA
    :param ScaleFactor: The scaling of each dimension in the PCA model
    :return: Output feature matrix dimensionally reduced by PCA
    """

    params_dict = {'X': X, 'Clusters': Clusters, 'Centering': Centering, 'ScaleFactor': ScaleFactor}
    return Matrix(X.sds_context,
        'pcaTransform',
        named_input_nodes=params_dict)
