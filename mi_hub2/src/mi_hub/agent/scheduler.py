"""外部計算資源へのジョブ投入層（Slurm / ローカルモック）。

DFT等の長時間計算を「投入→ポーリング→結果回収」の非同期モデルで扱う。
投入は必ず人間承認後（submit_dft_job は承認必須アクション）。
SlurmScheduler はログインノード上での直接実行と SSH 経由の両方に対応する。
"""

from __future__ import annotations

import shlex
import subprocess

#: Slurm→内部状態の対応。内部状態は pending / running / completed / failed / cancelled
_SLURM_STATE_MAP = {
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
    "CANCELLED": "cancelled",
}


class SchedulerError(RuntimeError):
    pass


class SchedulerGateway:
    """ジョブスケジューラの差し込み口。

    submit / status / cancel を実装すれば Slurm 以外（PBS 等）にも差し替え可能。
    """

    name: str = "scheduler"

    def submit(self, script: str, workdir: str, job_name: str) -> str:
        """ジョブスクリプトを投入し、スケジューラのジョブIDを返す。"""
        raise NotImplementedError

    def status(self, scheduler_job_id: str) -> str:
        """pending / running / completed / failed / cancelled のいずれかを返す。"""
        raise NotImplementedError

    def cancel(self, scheduler_job_id: str) -> None:
        raise NotImplementedError


class SlurmScheduler(SchedulerGateway):
    """Slurm（sbatch / squeue / sacct / scancel）によるジョブ投入。

    ssh_host を指定すると HPC ログインノードへ SSH 経由でコマンドを送る
    （鍵認証を前提とし、パスワードは扱わない）。
    """

    name = "slurm"

    def __init__(self, ssh_host: str | None = None, timeout_s: int = 60):
        self.ssh_host = ssh_host
        self.timeout_s = timeout_s

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if self.ssh_host:
            cmd = ["ssh", self.ssh_host, " ".join(shlex.quote(a) for a in args)]
        else:
            cmd = args
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=self.timeout_s, check=False)

    def submit(self, script: str, workdir: str, job_name: str) -> str:
        script_path = f"{workdir}/{job_name}.sbatch"
        if self.ssh_host:
            proc = subprocess.run(
                ["ssh", self.ssh_host,
                 f"mkdir -p {shlex.quote(workdir)} && cat > {shlex.quote(script_path)}"],
                input=script, capture_output=True, text=True,
                timeout=self.timeout_s, check=False)
            if proc.returncode != 0:
                raise SchedulerError(f"スクリプト転送失敗: {proc.stderr[-500:]}")
        else:
            import os
            os.makedirs(workdir, exist_ok=True)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)
        proc = self._run(["sbatch", "--parsable", "--chdir", workdir, script_path])
        if proc.returncode != 0:
            raise SchedulerError(f"sbatch 失敗: {proc.stderr[-500:]}")
        return proc.stdout.strip().split(";")[0]

    def status(self, scheduler_job_id: str) -> str:
        proc = self._run(["squeue", "-h", "-j", scheduler_job_id, "-o", "%T"])
        raw = proc.stdout.strip()
        if proc.returncode == 0 and raw:
            return _SLURM_STATE_MAP.get(raw, "running")
        # キューに無い場合は sacct で終了状態を確認
        proc = self._run(["sacct", "-n", "-X", "-j", scheduler_job_id,
                          "-o", "State", "--parsable2"])
        raw = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
        raw = raw.split(" ")[0]  # "CANCELLED by ..." 対策
        if not raw:
            raise SchedulerError(f"ジョブ状態を取得できません: {scheduler_job_id}")
        return _SLURM_STATE_MAP.get(raw, "failed")

    def cancel(self, scheduler_job_id: str) -> None:
        proc = self._run(["scancel", scheduler_job_id])
        if proc.returncode != 0:
            raise SchedulerError(f"scancel 失敗: {proc.stderr[-500:]}")


class LocalMockScheduler(SchedulerGateway):
    """HPC の無い環境で全経路を検証するためのモック。

    submit 時に bash で即時実行し、終了状態を保持する。
    Slurm と同じインターフェイスなので、承認・予算・ポーリングの
    ワークフローをローカルでそのまま試験できる。
    """

    name = "local_mock"

    def __init__(self, timeout_s: int = 600):
        self.timeout_s = timeout_s
        self._jobs: dict[str, str] = {}
        self._seq = 0

    def submit(self, script: str, workdir: str, job_name: str) -> str:
        import os
        os.makedirs(workdir, exist_ok=True)
        script_path = f"{workdir}/{job_name}.sbatch"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        self._seq += 1
        job_id = f"mock-{self._seq}"
        try:
            proc = subprocess.run(["bash", script_path], capture_output=True,
                                  text=True, timeout=self.timeout_s,
                                  cwd=workdir, check=False)
            with open(f"{workdir}/{job_name}.out", "w", encoding="utf-8") as f:
                f.write(proc.stdout)
            with open(f"{workdir}/{job_name}.err", "w", encoding="utf-8") as f:
                f.write(proc.stderr)
            self._jobs[job_id] = "completed" if proc.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            self._jobs[job_id] = "failed"
        return job_id

    def status(self, scheduler_job_id: str) -> str:
        if scheduler_job_id not in self._jobs:
            raise SchedulerError(f"ジョブ状態を取得できません: {scheduler_job_id}")
        return self._jobs[scheduler_job_id]

    def cancel(self, scheduler_job_id: str) -> None:
        self._jobs[scheduler_job_id] = "cancelled"


def make_sbatch_script(command: str, job_name: str,
                       partition: str | None = None,
                       nodes: int = 1, ntasks: int = 1,
                       time_limit: str = "01:00:00",
                       modules: list[str] | None = None) -> str:
    """VASP/QE 等の実行コマンドから sbatch スクリプトを組み立てる。"""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={job_name}.out",
        f"#SBATCH --error={job_name}.err",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines.append("")
    for mod in modules or []:
        lines.append(f"module load {mod}")
    lines += ["", command, ""]
    return "\n".join(lines)


def estimate_node_hours(nodes: int, time_limit: str) -> float:
    """time_limit（HH:MM:SS または D-HH:MM:SS）からノード時間上限を見積もる。"""
    days = 0
    if "-" in time_limit:
        d, time_limit = time_limit.split("-", 1)
        days = int(d)
    parts = [int(p) for p in time_limit.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return nodes * (days * 24 + h + m / 60 + s / 3600)


def build_scheduler(kind: str = "local_mock",
                    ssh_host: str | None = None) -> SchedulerGateway:
    if kind == "slurm":
        return SlurmScheduler(ssh_host=ssh_host)
    if kind == "local_mock":
        return LocalMockScheduler()
    raise ValueError(f"未対応のスケジューラ種別: {kind}")


def resolve_scheduler_from_env() -> SchedulerGateway:
    """環境変数からスケジューラを解決する。

    MI_HUB_SCHEDULER: slurm / local_mock（既定 local_mock）
    MI_HUB_SLURM_SSH_HOST: SSH 経由で Slurm を使う場合のホスト名
    """
    import os
    kind = os.environ.get("MI_HUB_SCHEDULER", "local_mock")
    return build_scheduler(kind, ssh_host=os.environ.get("MI_HUB_SLURM_SSH_HOST"))
