#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

# re-implementation of KernelExplainer to correctly handle DataFrames as input
# see also https://github.com/slundberg/shap/issues/2530

import gc
import logging

import numpy as np
import pandas as pd
import scipy as sp
from shap import KernelExplainer
from shap.utils._legacy import (
    DenseDataWithIndex,
    InstanceWithIndex,
    convert_to_instance_with_index,
)
from tqdm.auto import tqdm

log = logging.getLogger('shap')


class DataFrameData(DenseDataWithIndex):

    def __init__(self, data: pd.DataFrame, *args):
        super(DataFrameData, self).__init__(data, list(data.columns), data.index, data.index.name, *args)

    def convert_to_df(self):
        return self.data


class DataFrameInstance(InstanceWithIndex):

    def __init__(self, data: pd.DataFrame, i: int, group_display_values=None):
        super(DataFrameInstance, self).__init__(data.iloc[i:i + 1].copy(), list(data.columns),
                                                data.index, data.index.name, group_display_values)

    def convert_to_df(self):
        return self.x


class CustomKernelExplainer(KernelExplainer):
    """
    Custom kernel explainer that can handle DataFrames with mixed numeric and categorical data.
    See also https://github.com/slundberg/shap/issues/2530.
    """

    def __init__(self, model, data, keep_index: bool = True, **kwargs):
        if isinstance(data, pd.DataFrame) and keep_index:
            data = DataFrameData(data)
        super(CustomKernelExplainer, self).__init__(model, data, keep_index=keep_index, **kwargs)

    def shap_values(self, X, **kwargs):
        if isinstance(self.data, DenseDataWithIndex):
            column_name = self.data.group_names
            index_name = self.data.index_name
        else:
            column_name = index_name = None
        index_value = None

        # convert dataframes
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, pd.DataFrame) and not isinstance(self.data, DataFrameData):
            if self.keep_index:
                index_value = X.index.values
                index_name = X.index.name
                column_name = list(X.columns)
            X = X.values

        if not isinstance(X, pd.DataFrame):
            x_type = str(type(X))
            # if sparse, convert to lil for performance
            if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
                X = X.tolil()
            assert x_type.endswith("'numpy.ndarray'>") or sp.sparse.isspmatrix_lil(X), \
                'Unknown instance type: ' + x_type
            assert len(X.shape) == 1 or len(X.shape) == 2, 'Instance must have 1 or 2 dimensions!'
            assert not (self.keep_index and column_name is None)

        # single instance
        if len(X.shape) == 1:
            data = X.reshape((1, X.shape[0]))
            if self.keep_index:
                data = convert_to_instance_with_index(data, column_name, [0], index_name)
            explanation = self.explain(data, **kwargs)

            # vector-output
            s = explanation.shape
            if len(s) == 2:
                outs = [np.zeros(s[0]) for _ in range(s[1])]
                for j in range(s[1]):
                    outs[j] = explanation[:, j]
                return outs

            # single-output
            else:
                out = np.zeros(s[0])
                out[:] = explanation
                return out

        # explain the whole dataset
        elif len(X.shape) == 2:
            explanations = []
            for i in tqdm(range(X.shape[0]), disable=kwargs.get('silent', False)):
                if isinstance(X, pd.DataFrame):
                    data = DataFrameInstance(X, i)
                else:
                    data = X[i:i + 1]
                    if self.keep_index:
                        data = convert_to_instance_with_index(data, column_name,
                                                              index_value[i:i + 1] if index_value else [i], index_name)
                explanations.append(self.explain(data, **kwargs))
                if kwargs.get('gc_collect', False):
                    gc.collect()

            # vector-output
            s = explanations[0].shape
            if len(s) == 2:
                outs = [np.zeros((X.shape[0], s[0])) for _ in range(s[1])]
                for i in range(X.shape[0]):
                    for j in range(s[1]):
                        outs[j][i] = explanations[i][:, j]
                return outs

            # single-output
            else:
                out = np.zeros((X.shape[0], s[0]))
                for i in range(X.shape[0]):
                    out[i] = explanations[i]
                return out

    def explain(self, incoming_instance, **kwargs):
        if isinstance(self.data, DataFrameData):
            assert isinstance(incoming_instance, DataFrameInstance)
            if incoming_instance.group_display_values is None:
                incoming_instance.group_display_values = \
                    [incoming_instance.x.iloc[0, group[0]] if len(group) == 1 else '' for group in self.data.groups]
        else:
            assert not isinstance(incoming_instance, DataFrameInstance)
        return super(CustomKernelExplainer, self).explain(incoming_instance, **kwargs)

    def varying_groups(self, x) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            varying_indices = []
            for i in range(self.data.groups_size):
                for idx in self.data.groups[i]:
                    x_idx = x.iloc[0, idx]
                    if pd.isna(x_idx):
                        mismatch = self.data.data.iloc[:, idx].notna().any()
                    elif x.dtypes[idx].kind in 'fiub' and self.data.data.dtypes[idx].kind in 'fiub':
                        mismatch = not np.isclose(x_idx, self.data.data.iloc[:, idx], equal_nan=True).all()
                    else:
                        mismatch = (x_idx != self.data.data.iloc[:, idx]).any()
                    if mismatch:
                        varying_indices.append(i)
                        break
            return np.array(varying_indices)
        else:
            return super(CustomKernelExplainer, self).varying_groups(x)

    def allocate(self):
        if isinstance(self.data, DataFrameData):
            self.synth_data = pd.concat([self.data.data] * self.nsamples, axis=0, sort=False)
            self.maskMatrix = np.zeros((self.nsamples, self.M))
            self.kernelWeights = np.zeros(self.nsamples)
            self.y = np.zeros((self.nsamples * self.N, self.D))
            self.ey = np.zeros((self.nsamples, self.D))
            self.lastMask = np.zeros(self.nsamples)
            self.nsamplesAdded = 0
            self.nsamplesRun = 0
        else:
            super(CustomKernelExplainer, self).allocate()

    def addsample(self, x, m, w):
        offset = self.nsamplesAdded * self.N
        if isinstance(self.varyingFeatureGroups, list):
            for mj, group in zip(m, self.varyingFeatureGroups):
                if mj == 1.0:
                    for k in group:
                        if isinstance(self.synth_data, pd.DataFrame):
                            self.synth_data.iloc[offset:offset + self.N, k] = x.iloc[0, k]
                        else:
                            self.synth_data[offset:offset + self.N, k] = x[0, k]
        else:
            # for non-jagged numpy array we can significantly boost performance
            groups = self.varyingFeatureGroups[m == 1.0]
            if len(groups.shape) == 2:
                for group in groups:
                    if isinstance(self.synth_data, pd.DataFrame):
                        self.synth_data.iloc[offset:offset + self.N, group] = x.iloc[0, group]
                    else:
                        self.synth_data[offset:offset + self.N, group] = x[0, group]
            elif isinstance(self.synth_data, pd.DataFrame):
                self.synth_data.iloc[offset:offset + self.N, groups] = x.iloc[0, groups]
            else:
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0, groups]
                # In edge case where background is all dense but evaluation data
                # is all sparse, make evaluation data dense
                if sp.sparse.issparse(evaluation_data) and not sp.sparse.issparse(self.synth_data):
                    evaluation_data = evaluation_data.toarray()
                self.synth_data[offset:offset + self.N, groups] = evaluation_data
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        if isinstance(self.synth_data, pd.DataFrame):
            data = self.synth_data.iloc[self.nsamplesRun * self.N:self.nsamplesAdded * self.N]
        else:
            data = self.synth_data[self.nsamplesRun * self.N:self.nsamplesAdded * self.N]
            if self.keep_index:
                index = self.synth_data_index[self.nsamplesRun * self.N:self.nsamplesAdded * self.N]
                index = pd.DataFrame(index, columns=[self.data.index_name])
                data = pd.DataFrame(data, columns=self.data.group_names)
                data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
                if self.keep_index_ordered:
                    data = data.sort_index()
        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        self.y[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1
