from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Library(StrEnum):
    MIPLIB_BENCHMARK = "miplib_benchmark"
    MIPLIB_COLLECTION = "miplib_collection"
    MINLPLIB = "minlplib"
    QPLIB = "qplib"

    @property
    def is_miplib(self) -> bool:
        return self in {Library.MIPLIB_BENCHMARK, Library.MIPLIB_COLLECTION}

    @property
    def family(self) -> str:
        return "miplib" if self.is_miplib else self.value

    @property
    def tag(self) -> str:
        if not self.is_miplib:
            msg = "Tag is only available for MIPLIB libraries."
            raise ValueError(msg)
        if self == Library.MIPLIB_BENCHMARK:
            return "benchmark"
        elif self == Library.MIPLIB_COLLECTION:
            return "collection"

    @property
    def website(self) -> str:
        if self.is_miplib:
            return "https://miplib.zib.de/"
        elif self == Library.MINLPLIB:
            return "http://www.minlplib.org/"
        elif self == Library.QPLIB:
            return "http://www.qplib.de/"

    @property
    def table_url(self) -> str:
        html = "instances.html"
        if self.is_miplib:
            html = f"tag_{self.tag}.html"
        return f"{self.website}{html}"

    @property
    def download_url(self) -> str:
        if self.is_miplib:
            return f"{self.website}" + "WebData/instances/" + "{name}.{fmt}"
        elif self == Library.MINLPLIB:
            return f"{self.website}" + "{fmt}/{name}.{fmt}"
        elif self == Library.QPLIB:
            return f"{self.website}" + "{fmt}/QPLIB_{name}.{fmt}"


class Status(StrEnum):
    EASY = "easy"
    HARD = "hard"
    OPEN = "open"
    CLOSED = "closed"


class OptimizationStatus(StrEnum):
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    FEASIBLE = "feasible"
    OPTIMAL = "optimal"
    UNKNOWN = "unknown"


class ProblemType(StrEnum):
    LP = "LP"
    BLP = "BLP"
    ILP = "ILP"
    MBLP = "MBLP"
    MILP = "MILP"


class Format(StrEnum):
    MPS = "mps"
    LP = "lp"


@dataclass
class Instance:
    library: Library
    name: str
    problem_type: ProblemType
    path: Path | None
    status: Status | None
    optimization_status: OptimizationStatus
    primal: float | None
    dual: float | None
    n_vars: int
    n_bins: int
    n_ints: int
    n_conts: int
    n_cons: int
    n_nz: int
    n_sos: int | None = None
    n_semi: int | None = None
    q0_density: float | None = None
    q0_ev_prob: float | None = None
    n_quads: int | None = None
    obj_type: str | None = None
    var_type: str | None = None
    cons_type: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    formats: list[Format] = field(default_factory=list)

    def __post_init__(self):
        self.library = Library(self.library)
        self.problem_type = ProblemType(self.problem_type)
        self.optimization_status = OptimizationStatus(self.optimization_status)
        if self.status is not None:
            self.status = Status(self.status)

    def remove(self):
        if self.path is not None and self.path.exists():
            self.path.unlink()

    def __str__(self) -> str:
        attrs = {
            "Instance": self.name,
            "Problem Type": self.problem_type,
            "Local path": self.path or "-",
            "Status": self.status or "-",
            "Optimization Status": self.optimization_status,
            "Primal Bound": self.primal if self.primal is not None else "-",
            "Dual Bound": self.dual if self.dual is not None else "-",
            "Variables": (
                f"{self.n_vars}"
                + (f" ({self.n_bins} binary)" if self.n_bins > 0 else "")
                + (f" ({self.n_ints} integer)" if self.n_ints > 0 else "")
            ),
            "Constraints": (
                f"{self.n_cons}"
                + (
                    f" ({self.n_quads} quadratic)"
                    if self.n_quads is not None and self.n_quads > 0
                    else ""
                )
                + (
                    f" ({self.n_sos} SOS)"
                    if self.n_sos is not None and self.n_sos > 0
                    else ""
                )
                + (
                    f" ({self.n_semi} semi)"
                    if self.n_semi is not None and self.n_semi > 0
                    else ""
                )
            ),
            "Non-zeroes": self.n_nz,
        }

        if self.q0_density is not None:
            attrs.update(
                {
                    "Q0 Density": f"{self.q0_density}%",
                    "Q0 EV Density": f"{self.q0_ev_prob}%",
                    "Objective Type": self.obj_type or "-",
                    "Variables Type": self.var_type or "-",
                    "Constraints Type": self.cons_type or "-",
                }
            )

        if self.tags:
            attrs["Tags"] = ", ".join(self.tags)

        if self.formats:
            attrs["Formats"] = ", ".join(self.formats)

        max_key_length = max(len(key) for key in attrs)
        description = "\n".join(
            f"{key.ljust(max_key_length)}: {value}"
            for key, value in attrs.items()
        )
        return description
