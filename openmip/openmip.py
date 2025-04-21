import gzip
import io
import re
import shlex
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
import requests
from filelock import FileLock, Timeout
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .instance import (
    Format,
    Instance,
    Library,
    OptimizationStatus,
    ProblemType,
    Status,
)


@dataclass(frozen=True)
class Columns:
    NAME: str = "name"
    STATUS: str = "status"
    N_VARS: str = "n_vars"
    N_BINS: str = "n_bins"
    N_INTS: str = "n_ints"
    N_CONTS: str = "n_conts"
    N_CONS: str = "n_cons"
    N_NZ: str = "n_nz"
    N_SOS: str = "n_sos"
    N_SEMI: str = "n_semi"
    N_QUADS: str = "n_quads"
    TYPE: str = "type"
    Q0_DENSITY: str = "q0_density"
    Q0_EV_PROB: str = "q0_ev_prob"
    GROUP: str = "group"
    TAGS: str = "tags"
    PRIMAL: str = "primal"
    FEASIBLE: str = "feasible"
    OPTIMIZATION_STATUS: str = "optimization_status"


COLUMNS = Columns()


class OpenMIP:
    _LOCK_TIMEOUT = 60

    cache_path: Path
    console: Console
    progress: Progress

    def __init__(
        self,
        cache_path: str | Path = ".openmip",
        *,
        verbose: bool = True,
    ) -> None:
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.console = Console(quiet=not verbose)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            disable=not verbose,
        )

    def load(
        self,
        library: Library,
        *filters: str,
        refresh: bool = False,
    ) -> list[Instance]:
        library = Library(library)
        filters = tuple(filters or ())

        csv_path = self._locate_cache_path(library) / "instances.csv"
        lock_path = csv_path.with_suffix(".lock")
        try:
            with FileLock(str(lock_path), timeout=self._LOCK_TIMEOUT):
                if refresh or not csv_path.exists():
                    table = self._download_table(library)
                    table = self._preprocess_table(table, library)
                    self._write_csv(table, csv_path)
                else:
                    table = pd.read_csv(csv_path)
        except Timeout:
            msg = (
                "[yellow]Another OpenMIP process is busy;"
                " using existing cache.[/yellow]"
            )
            self.console.print(msg, end="")
            table = pd.read_csv(csv_path)

        if filters:
            query = " & ".join(f for f in filters)
            table = table.query(query)

        return self._build_instances(table, library)

    def download(
        self,
        instance: Instance,
        *,
        fmt: Format | None = None,
        refresh: bool = False,
        subdir: str | None = None,
    ) -> None:
        library = instance.library
        if fmt is None:
            if not instance.formats:
                msg = "No file format specified and instance has no formats."
                raise ValueError(msg)
            fmt = instance.formats[0]
        if fmt not in instance.formats:
            msg = f"Instance {instance.name} does not support {fmt}."
            raise ValueError(msg)

        if instance.path is not None and instance.path.suffix != f".{fmt}":
            instance.remove()
            instance.path = None

        if instance.path is None:
            path = self.cache_path / library.family
            if subdir is not None:
                path /= subdir
            path = path / f"{fmt}" / f"{instance.name}.{fmt}"
            instance.path = path

        if instance.path.exists() and not refresh:
            msg = (
                f"[yellow]Instance {instance.name} already exists;"
                " skipping.[/yellow]"
            )
            self.console.print(msg, end="")
            return

        instance.path.parent.mkdir(parents=True, exist_ok=True)

        tmp = instance.path.with_suffix(".tmp")
        url = library.download_url.format(name=instance.name, fmt=fmt)
        if library.is_miplib:
            url += ".gz"
        else:
            msg = f"Library {library} not supported yet."
            raise NotImplementedError(msg)
        with self.progress:
            desc = (
                f"[cyan]Downloading [bold]{instance.name}[/bold] "
                f"instance...[/cyan]"
            )
            task = self.progress.add_task(desc, total=None)
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with tmp.open("wb") as writer:
                    writer.write(response.content)
            finally:
                self.progress.update(task, completed=1)
                self.progress.remove_task(task)
                msg = f"[green]Downloaded {instance.name}.[/green]"
                self.console.print(msg)
        if library.is_miplib:
            with (
                gzip.open(tmp, "rb") as reader,
                instance.path.open("wb") as writer,
            ):
                writer.write(reader.read())
        else:
            tmp.replace(instance.path)
        tmp.unlink(missing_ok=True)

    def _locate_cache_path(self, library: Library) -> Path:
        if library.is_miplib:
            subdir = library.tag
            path = self.cache_path / library.family / subdir
        else:
            path = self.cache_path / library.family
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _download_table(self, library: Library) -> pd.DataFrame:
        with self.progress:
            desc = (
                f"[cyan]Downloading [bold]{library.name}[/bold] table...[/cyan]"
            )
            task = self.progress.add_task(desc, total=None)
            try:
                response = requests.get(library.table_url, timeout=30)
                response.raise_for_status()
                reader = io.StringIO(response.text)
                tables = pd.read_html(reader)
            finally:
                self.progress.update(task, completed=1)
                self.progress.remove_task(task)

        if not tables:
            raise RuntimeError(f"No tables found at {library.table_url}")
        return tables[0]

    def _preprocess_table(
        self,
        table: pd.DataFrame,
        library: Library,
    ) -> pd.DataFrame:
        if library.is_miplib:
            table = self._preprocess_miplib(table)
        else:
            msg = f"Library {library} not supported yet."
            raise NotImplementedError(msg)
        return table

    @staticmethod
    def _write_csv(table: pd.DataFrame, path: Path) -> None:
        tmp_path = path.with_suffix(".tmp")
        table.to_csv(tmp_path, index=False)
        tmp_path.replace(path)

    @staticmethod
    def _preprocess_miplib(table: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "Instance": COLUMNS.NAME,
            "Status": COLUMNS.STATUS,
            "Variables": COLUMNS.N_VARS,
            "Binaries": COLUMNS.N_BINS,
            "Integers": COLUMNS.N_INTS,
            "Continuous": COLUMNS.N_CONTS,
            "Constraints": COLUMNS.N_CONS,
            "Nonz.": COLUMNS.N_NZ,
            "Group": COLUMNS.GROUP,
            "Objective": COLUMNS.PRIMAL,
            "Tags": COLUMNS.TAGS,
        }
        table = table.rename(columns=mapping)[mapping.values()]

        def _mip_type(row: pd.Series) -> str:
            n_vars = int(row[COLUMNS.N_VARS])
            n_bins = int(row[COLUMNS.N_BINS])
            n_ints = int(row[COLUMNS.N_INTS])
            n_conts = int(row[COLUMNS.N_CONTS])
            if n_conts == n_vars:
                return ProblemType.LP
            if n_bins == n_vars:
                return ProblemType.BLP
            if n_ints == n_vars:
                return ProblemType.ILP
            if n_ints == 0:
                return ProblemType.MBLP
            return ProblemType.MILP

        table[COLUMNS.TYPE] = table.apply(_mip_type, axis=1)

        unbounded = table[COLUMNS.PRIMAL].str.contains(
            OptimizationStatus.UNBOUNDED,
            flags=re.IGNORECASE,
            na=False,
        )
        infeasible = table[COLUMNS.PRIMAL].str.contains(
            OptimizationStatus.INFEASIBLE,
            flags=re.IGNORECASE,
            na=False,
        )
        is_open = table[COLUMNS.STATUS].str.contains(
            Status.OPEN,
            flags=re.IGNORECASE,
            na=True,
        )
        optimal = ~(is_open | unbounded | infeasible)
        starred = table[COLUMNS.PRIMAL].str.endswith("*", na=False)
        feasible = is_open & starred
        has_primal = optimal | feasible
        table.loc[has_primal, COLUMNS.PRIMAL] = (
            table.loc[has_primal, COLUMNS.PRIMAL]
            .str.rstrip("*")
            .apply(pd.to_numeric, errors="coerce")
        )
        table.loc[~has_primal, COLUMNS.PRIMAL] = pd.NA
        column = COLUMNS.OPTIMIZATION_STATUS
        table[column] = OptimizationStatus.UNKNOWN
        table.loc[optimal, column] = OptimizationStatus.OPTIMAL
        table.loc[unbounded, column] = OptimizationStatus.UNBOUNDED
        table.loc[infeasible, column] = OptimizationStatus.INFEASIBLE
        table.loc[feasible, column] = OptimizationStatus.FEASIBLE

        return table

    def _build_instances(
        self,
        table: pd.DataFrame,
        library: Library,
    ) -> list[Instance]:
        if library.is_miplib:
            builder = partial(self._build_miplib_instance, library=library)
        else:
            msg = f"Library {library} not supported yet."
            raise NotImplementedError(msg)
        return [builder(row) for _, row in table.iterrows()]

    def _build_miplib_instance(
        self,
        series: pd.Series,
        *,
        library: Library,
    ) -> Instance:
        tags = (
            shlex.split(str(series[COLUMNS.TAGS]))
            if pd.notna(series[COLUMNS.TAGS])
            else []
        )
        tags = [tag.strip() for tag in tags if tag.strip()]
        objective = (
            series[COLUMNS.PRIMAL] if pd.notna(series[COLUMNS.PRIMAL]) else None
        )
        return Instance(
            library=library,
            name=str(series[COLUMNS.NAME]),
            path=None,
            problem_type=str(series[COLUMNS.TYPE]),
            status=str(series[COLUMNS.STATUS]),
            optimization_status=str(series[COLUMNS.OPTIMIZATION_STATUS]),
            primal=objective,
            dual=objective,
            n_vars=int(series[COLUMNS.N_VARS]),
            n_bins=int(series[COLUMNS.N_BINS]),
            n_ints=int(series[COLUMNS.N_INTS]),
            n_conts=int(series[COLUMNS.N_CONTS]),
            n_cons=int(series[COLUMNS.N_CONS]),
            n_nz=int(series[COLUMNS.N_NZ]),
            group=series[COLUMNS.GROUP],
            tags=tags,
            formats=[Format.MPS],
        )
