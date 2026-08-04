"""Thin external-controller client boundary.

No estimator, result conversion, SSH, or SLURM implementation belongs in this
package. Those contracts are owned by ``posetestbot-cluster``.
"""

from .client import ClusterClientError, ClusterControllerClient

__all__ = ["ClusterClientError", "ClusterControllerClient"]
