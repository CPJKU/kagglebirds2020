# -*- coding: utf-8 -*-

"""
Definitions to plug together for an experiment.

Author: Jan Schlüter
"""
from .datasets import get_dataset, get_dataloader
from .models import get_model
from .metrics import get_metrics, get_loss_from_metrics
from .optimizers import get_optimizer
