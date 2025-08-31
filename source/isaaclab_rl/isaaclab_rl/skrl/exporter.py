# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_jit(policy: object, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""

class _StandalonePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.net_container = policy.net_container

    def forward(self, x):
        return torch.nn.functional.tanh(self.net_container(x))

class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy):
        super().__init__()
        # copy policy parameters
        if hasattr(policy, "policy"):
            self.actor = _StandalonePolicy(copy.deepcopy(policy.policy))
        else:
            raise ValueError("Policy does not have an actor module.")

    def forward(self, x):
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, verbose=False):
        super().__init__()
        self.verbose = verbose
        # copy policy parameters
        if hasattr(policy, "policy"):
            self.actor = _StandalonePolicy(copy.deepcopy(policy.policy))
        else:
            raise ValueError("Policy does not have an actor module.")

    def forward(self, x):
        return self.actor(x)

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor.net_container[0].in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
        )
