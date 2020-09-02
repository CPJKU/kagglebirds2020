"""
Multiple-input, multiple-output model, with its parts defined via custom_cnn.
Provides create().

Author: Jan Schlüter
"""
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import PickDictKey, PutDictKey, ReceptiveField
from .custom_cnn import custom_cnn


class DAG(nn.Module):
    """
    Generic directed acyclic graph of submodules.
    """
    def __init__(self):
        super(DAG, self).__init__()
        self.inputs = dict()
        self.outputs = set()
        self.extra_outputs = set()
        self.lifetimes = dict()

    def append(self, module, to=None, name=None, as_output=False):
        """
        Register `module` to process the output(s) of `to` and make it
        available under the given `name`.

        If `to` is not given, `module` will be attached to the module added
        most recently, or to the input.
        If `to` is a string, it refers to the output of the module added under
        that name, or the name of an input.
        If `to` is a collection (tuple, list, dict, or set), the referred
        outputs or inputs will be passed as a collection of the same structure,
        except that sets will be replaced by dictionaries using the registered
        module names or input names as keys.

        If `name` is not given, `module` will be named with a number (the
        number of previously registered modules converted to a string).

        If `as_output` is true-ish, the output of `module` will be registered
        as an extra output of the DAG to be returned on a forward() call even
        if it is not a topmost module.

        Modifies the DAG in-place and returns it, to allow chaining calls.
        """
        idx = len(self._modules)
        # choose module to attach to if not given
        if to is None and self._modules:
            to = next(reversed(self._modules))
        # generate name if not given
        if name is None:
            name = str(idx)
        # register as submodule
        self.add_module(name, module)
        # shortcut: allow {name1, name2} to mean {name1: name1, name2: name2}
        if isinstance(to, set):
            to = {k: k for k in to}
        # register required inputs
        self.inputs[name] = to
        # take note how long inputs are required for the forward pass,
        # and take note that they are not default output layers
        if isinstance(to, str):
            to = (to,)
        if to is not None:
            for k in to:
                self.lifetimes[k] = idx
                self.outputs.discard(k)
        # register as a default output
        self.outputs.add(name)
        # register as extra output, if requested
        if as_output:
            self.extra_outputs.add(name)
        # return the DAG
        return self

    def forward(self, data):
        data = dict(data)  # create a modifiable copy; XXX: requires dict input

        def take_data(name, idx):
            if self.lifetimes.get(name, -1) == idx:
                return data.pop(name)
            else:
                return data[name]

        outputs = OrderedDict()
        output_keys = self.outputs | self.extra_outputs
        for idx, (name, module) in enumerate(self._modules.items()):
            # collect inputs
            inputs = self.inputs[name]
            if inputs is None:
                inputs = data
            elif isinstance(inputs, str):
                inputs = take_data(inputs, idx)
            elif isinstance(inputs, (list, tuple)):
                inputs = type(inputs)(take_data(k, idx) for k in inputs)
            elif isinstance(inputs, dict):
                inputs = type(inputs)((k, take_data(k, idx)) for k in inputs)
            # pass through module
            output = module(inputs)
            # save outputs
            data[name] = output
            if name in output_keys:
                outputs[name] = output
        return outputs


def create(cfg, shapes, dtypes, num_classes):
    """
    Instantiates a Model for the given data shapes and dtypes.

    Controlled by the following configuration flags:
    - model.graph: defines the mapping of input names to submodels and their
        output names, in topological order. Must be a string of the format:
        name:inputs->output;name:inputs->output;...
        Where "name" refers to a custom_cnn submodule defined at
        "model.name.arch", "inputs" defines the name(s) of required inputs as
        specified below, and "output" defines the name to give to the produced
        output.
        "inputs" can define one or more input names. If a single name, it will
        be fed to the submodel as a tensor. If separated by commas, they will
        be fed to the submodel as a dictionary. If separated by plus signs,
        they will be added and fed to the submodel as a tensor. If separated by
        asterisks, they will be multiplied and fed to the submodel as a tensor.
        If separated by pipes, they will be concatenated along the first
        dimension and fed to the submodel as a tensor.
        Output names preceded by an underscore will be treated as intermediate
        results and not included in the final output of the model.
    - model.<name>.arch: For each name given in the graph definition, the
        architecture is to be specified in the format expected by custom_cnn.
    """
    num_outputs = 1 if num_classes == 2 else num_classes
    model = DAG()
    channels = {k: shape[0] for k, shape in shapes.items() if len(shape)}
    receptive_fields = {k: ReceptiveField() for k in shapes.keys()}

    graph = cfg['model.graph'].split(';')
    for node in graph:
        name, edge = node.split(':', 1)
        inputs, output = edge.split('->', 1)
        if any(c in inputs for c in ',+*|'):
            raise NotImplementedError("only single inputs for now")
        if ',' in output:
            raise ValueError("invalid output name %r (forgot semicolon?)" %
                             output)
        specification = cfg['model.%s.arch' % name]
        specification = specification.replace('C', str(num_outputs))
        submodule = custom_cnn(channels[inputs], specification,
                               input_name=None, output_name=None)
        receptive_fields[output] = (receptive_fields[inputs] *
                                    submodule.receptive_field)
        channels[output] = submodule.out_channels
        model.append(submodule, inputs, output, as_output=output[0] != '_')

    model.receptive_field = sum(receptive_fields[o]
                                for o in model.outputs | model.extra_outputs)
    return model
