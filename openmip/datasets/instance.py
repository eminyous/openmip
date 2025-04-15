from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .status import MINLPLIBStatus, MIPLIBStatus


@dataclass
class BaseInstance(ABC):
    name: str
    path: Path
    n_vars: int
    n_bins: int
    n_ints: int
    n_cons: int
    n_constrs: int
    n_nz: int

    def get_base_info(self) -> dict[str, str]:
        return {
            "Instance": self.name,
            "Local path": str(self.path),
            "Number of variables": str(self.n_vars),
            "Number of binary variables": str(self.n_bins),
            "Number of integer variables": str(self.n_ints),
            "Number of continuous variables": str(self.n_cons),
            "Number of constraints": str(self.n_constrs),
            "Number of non-zeros": str(self.n_nz),
        }

    def get_info(self) -> dict[str, str]:
        info = self.get_base_info()
        extended_info = self.get_extended_info()
        info.update(extended_info)
        return info

    @abstractmethod
    def get_extended_info(self) -> dict[str, str]:
        raise NotImplementedError

    def __str__(self) -> str:
        info = self.get_info()
        max_key_len = max(len(k) for k in info)
        res = ""
        for k, v in info.items():
            res += f"{k:<{max_key_len}}: {v}\n"
        return res.strip()


@dataclass
class BaseNonLinearInstance(ABC, BaseInstance):
    formats: list[str]
    is_relax_convex: bool


@dataclass
class MIPLIBInstance(BaseInstance):
    group: str | None
    feasible: bool
    objective: float
    tags: list[str]
    status: MIPLIBStatus

    def get_extended_info(self) -> dict[str, str]:
        info = {
            "Group": self.group if self.group is not None else "-",
            "Feasible": str(self.feasible),
            "Objective": str(self.objective) if self.feasible else "-",
            "Tags": ", ".join(self.tags) if self.tags else "-",
            "Status": self.status.value,
        }
        return info


@dataclass
class QPLIBInstance(BaseNonLinearInstance):
    obj_type: str
    q0_density: float
    q0_ev_density: float
    var_type: str
    constr_type: str
    n_quad_constrs: int

    def get_extended_info(self) -> dict[str, str]:
        info = {
            "Convexity": str(self.is_relax_convex),
            "Objective type": self.obj_type,
            "Q0 density": str(self.q0_density),
            "Q0 eigen value density": str(self.q0_ev_density),
            "Variable type": self.var_type,
            "Constraint type": self.constr_type,
            "Number of quadratic constraints": str(self.n_quad_constrs),
        }
        return info


@dataclass
class MINLPLIBInstance(BaseNonLinearInstance):
    prob_type: str
    n_sos_constrs: int
    n_semi_constrs: int
    dual: float
    primal: float
    status: MINLPLIBStatus

    def get_extended_info(self) -> dict[str, str]:
        info = {
            "Convexity": str(self.is_relax_convex),
            "Problem type": self.prob_type,
            "Number of second order cone constraints": str(self.n_sos_constrs),
            "Number of semi-continuous constraints": str(self.n_semi_constrs),
            "Dual objective value": str(self.dual)
            if self.dual != float("-inf")
            else "-",
            "Primal objective value": str(self.primal)
            if self.primal != float("-inf")
            else "-",
            "Status": self.status.value,
        }
        return info
