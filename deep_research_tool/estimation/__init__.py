"""
Fermi Estimation module for Deep Research Tool.

Provides Fermi estimation capabilities to derive market size, demand,
or other quantities that aren't directly available in evidence.
"""

from .fermi_estimator import FermiEstimator, FermiEstimationConfig, FermiEstimationResult
from .decomposer import Decomposer, DecompositionTree, TreeNode, NodeOperation
from .assumptions import AssumptionManager, Assumption, AssumptionSource
from .calculator import Calculator, ScenarioResult, SensitivityAnalysis
from .validator import Validator, ValidationResult, DomainPrior, DomainPriorProvider

__all__ = [
    "FermiEstimator",
    "FermiEstimationConfig",
    "FermiEstimationResult",
    "Decomposer",
    "DecompositionTree",
    "TreeNode",
    "NodeOperation",
    "AssumptionManager",
    "Assumption",
    "AssumptionSource",
    "Calculator",
    "ScenarioResult",
    "SensitivityAnalysis",
    "Validator",
    "ValidationResult",
    "DomainPrior",
    "DomainPriorProvider",
]
