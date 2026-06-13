"""
Job scheduler script generator — PBS / Slurm / local.

Generates ready-to-submit job scripts with configurable resources,
module loading, and VASP execution commands.
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


def _load_scheduler_defaults() -> dict:
    cfg_path = Path(__file__).parent / "config" / "defaults.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("scheduler", {})


def generate_job_script(
    output_dir: Path,
    scheduler: str = "slurm",
    job_name: str = "t2vasp",
    nodes: int = 1,
    ncpus: Optional[int] = None,
    walltime: Optional[str] = None,
    queue: str = "",
    modules: Optional[List[str]] = None,
    vasp_cmd: Optional[str] = None,
) -> Path:
    """Generate a job submission script.

    Parameters
    ----------
    output_dir : Path
        Directory where the script will be written.
    scheduler : str
        One of "slurm", "pbs", "local".
    job_name : str
        Job name.
    nodes : int
        Number of nodes.
    ncpus : int, optional
        CPUs per node.  Defaults from config.
    walltime : str, optional
        Walltime string (HH:MM:SS).
    queue : str
        Queue/partition name.
    modules : list of str, optional
        Module names to load.
    vasp_cmd : str, optional
        VASP execution command.

    Returns
    -------
    Path
        Path to the generated script.
    """
    defaults = _load_scheduler_defaults()

    ncpus = ncpus or defaults.get("ncpus_per_node", 16)
    walltime = walltime or defaults.get("walltime", "24:00:00")
    modules = modules if modules is not None else defaults.get("modules", [])
    vasp_cmd = vasp_cmd or defaults.get("vasp_command",
                                        f"mpirun -np {ncpus} vasp_std")

    if scheduler == "slurm":
        return _write_slurm(output_dir, job_name, nodes, ncpus, walltime,
                            queue, modules, vasp_cmd)
    elif scheduler == "pbs":
        return _write_pbs(output_dir, job_name, nodes, ncpus, walltime,
                          queue, modules, vasp_cmd)
    elif scheduler == "local":
        return _write_local(output_dir, vasp_cmd)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler!r}. "
                         f"Choose from: slurm, pbs, local")


def _write_slurm(
    output_dir: Path,
    job_name: str,
    nodes: int,
    ncpus: int,
    walltime: str,
    queue: str,
    modules: List[str],
    vasp_cmd: str,
) -> Path:
    total_cpus = nodes * ncpus
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -N {nodes}",
        f"#SBATCH -n {total_cpus}",
        f"#SBATCH --ntasks-per-node={ncpus}",
        f"#SBATCH -t {walltime}",
        f"#SBATCH -o {job_name}_%j.out",
        f"#SBATCH -e {job_name}_%j.err",
    ]
    if queue:
        lines.append(f"#SBATCH -p {queue}")
    lines.append("")

    # Module loading
    if modules:
        lines.append("# Load modules")
        for mod in modules:
            lines.append(f"module load {mod}")
        lines.append("")

    lines.extend([
        "cd $SLURM_SUBMIT_DIR",
        "",
        "# Generate POTCAR if not present",
        'if [ ! -f "POTCAR" ]; then',
        '    echo "Generating POTCAR..."',
        "    bash make_potcar.sh",
        "fi",
        "",
        f'echo "Starting VASP: {job_name}"',
        f"echo \"Command: {vasp_cmd}\"",
        f"{vasp_cmd} > vasp.log 2>&1",
        "",
        "# Check convergence",
        'if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
        '    echo "Calculation converged successfully."',
        "else",
        '    echo "WARNING: Calculation may not have converged."',
        "fi",
        "",
    ])

    path = output_dir / "job_slurm.sh"
    path.write_text("\n".join(lines))
    path.chmod(0o755)
    logger.info("Wrote Slurm script: %s", path)
    return path


def _write_pbs(
    output_dir: Path,
    job_name: str,
    nodes: int,
    ncpus: int,
    walltime: str,
    queue: str,
    modules: List[str],
    vasp_cmd: str,
) -> Path:
    lines = [
        "#!/bin/bash",
        f"#PBS -N {job_name}",
        f"#PBS -l select={nodes}:ncpus={ncpus}:mpiprocs={ncpus}",
        f"#PBS -l walltime={walltime}",
        "#PBS -j oe",
    ]
    if queue:
        lines.append(f"#PBS -q {queue}")
    lines.append("")

    if modules:
        lines.append("# Load modules")
        for mod in modules:
            lines.append(f"module load {mod}")
        lines.append("")

    lines.extend([
        "cd $PBS_O_WORKDIR",
        "",
        "# Generate POTCAR if not present",
        'if [ ! -f "POTCAR" ]; then',
        '    echo "Generating POTCAR..."',
        "    bash make_potcar.sh",
        "fi",
        "",
        f'echo "Starting VASP: {job_name}"',
        f"{vasp_cmd} > vasp.log 2>&1",
        "",
        'if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
        '    echo "Calculation converged successfully."',
        "else",
        '    echo "WARNING: Calculation may not have converged."',
        "fi",
        "",
    ])

    path = output_dir / "job_pbs.sh"
    path.write_text("\n".join(lines))
    path.chmod(0o755)
    logger.info("Wrote PBS script: %s", path)
    return path


def _write_local(output_dir: Path, vasp_cmd: str) -> Path:
    lines = [
        "#!/bin/bash",
        "# Local execution script (auto-generated by t2vasp)",
        "",
        "# Generate POTCAR if not present",
        'if [ ! -f "POTCAR" ]; then',
        '    echo "Generating POTCAR..."',
        "    bash make_potcar.sh",
        "fi",
        "",
        'echo "Running VASP locally..."',
        f"{vasp_cmd} > vasp.log 2>&1",
        "",
        'if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
        '    echo "Calculation converged successfully."',
        "else",
        '    echo "WARNING: Calculation may not have converged."',
        "fi",
        "",
    ]

    path = output_dir / "run_local.sh"
    path.write_text("\n".join(lines))
    path.chmod(0o755)
    logger.info("Wrote local run script: %s", path)
    return path
