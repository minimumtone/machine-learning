#!/usr/bin/env python3
"""
ISIF=7 SQS計算の未収束ケースを検出し、自動再計算するスクリプト。

Phase 1 (--detect): vasp.log / OUTCARをスキャンして再計算が必要なケースを検出
Phase 2 (--rerun):  CONTCAR→POSCAR、INCAR修正、mpirun で再計算実行

検出する問題:
  1. ZBRENT エラー (can not reach accuracy / fatal error in bracketing)
  2. NSW上限到達（イオンステップ = NSW、まだ収束していない）
  3. OUTCARに "reached required accuracy" がない（未収束）
  4. CONTCAR が空 or 存在しない（計算未完了）
  5. Ω_sfアウトライア（|Ω_sf| > threshold）

使い方:
    # 検出のみ（ドライラン）
    python detect_rerun_needed.py /path/to/BCC_SQS_ISIF7

    # 検出 + 再計算実行（8コア×4ジョブ並列）
    python detect_rerun_needed.py /path/to/BCC_SQS_ISIF7 --rerun

    # オプション
    --ncore 8           MPI並列コア数 (default: 8)
    --max-jobs 4        同時実行ジョブ数 (default: 4)
    --nsw 300           再計算のNSW (default: 300)
    --potim VALUE       再計算のPOTIM (default: auto — ZBRENTエラー時に調整)
    --ibrion {1,2}      再計算のIBRION (default: auto — ZBRENT時に1へ切替)
    --encut VALUE        再計算のENCUT [eV] (default: auto — 未収束時に600へ増加)
    --addgrid            ADDGRID=.TRUE.を追加（FFTグリッド補間精度向上）
    --omega-threshold 0.5  |Ω_sf|アウトライア閾値 (default: 0.5)

環境変数:
    VASPBIN  VASPバイナリのパス (必須、--rerun時)
"""

import os
import re
import sys
import glob
import time
import shutil
import signal
import argparse
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


# =====================================================================
# Directory parsing
# =====================================================================
def parse_dirname(dirname):
    """Parse directory name like 'Ag8Al8' or 'Fe4Co4Ni4Cr4' into element list.

    Returns list of (element, count) tuples, or None if unparseable.
    For 2-element dirs: [(elA, cA), (elB, cB)]
    For N-element dirs: [(el1, c1), (el2, c2), ...]
    """
    pairs = re.findall(r'([A-Z][a-z]?)(\d+)', dirname)
    if not pairs:
        return None
    # Verify the pattern accounts for the entire dirname
    reconstructed = ''.join(el + cnt for el, cnt in pairs)
    if reconstructed != dirname:
        return None
    return [(el, int(cnt)) for el, cnt in pairs]


# =====================================================================
# Convergence checks
# =====================================================================
def check_vasp_log(calc_dir):
    """Check vasp.log for convergence issues. Returns list of issue strings."""
    issues = []

    # Find log file
    vasp_log = None
    for name in ['vasp.log', 'stdout']:
        p = os.path.join(calc_dir, name)
        if os.path.isfile(p):
            vasp_log = p
            break
    if vasp_log is None:
        for p in glob.glob(os.path.join(calc_dir, 'slurm-*.out')):
            vasp_log = p
            break
    if vasp_log is None:
        issues.append("NO_VASP_LOG")
        return issues

    try:
        with open(vasp_log, 'r', errors='replace') as f:
            content = f.read()
    except Exception as e:
        issues.append(f"READ_ERROR: {e}")
        return issues

    if 'ZBRENT: can not reach accuracy' in content:
        issues.append("ZBRENT_ACCURACY")
    if 'ZBRENT: fatal error in bracketing' in content:
        issues.append("ZBRENT_FATAL")
    if 'please rerun with smaller EDIFF' in content:
        issues.append("EDIFF_RERUN")
    if 'VERY BAD NEWS!' in content:
        issues.append("VERY_BAD_NEWS")
    if 'SGRCON: ERROR' in content:
        issues.append("SGRCON_ERROR")
    if 'internal error in subroutine PRICEL' in content:
        issues.append("PRICEL_ERROR")

    # Count ionic steps
    ionic_steps = re.findall(r'^\s+(\d+)\s+F=', content, re.MULTILINE)
    if ionic_steps:
        max_step = max(int(s) for s in ionic_steps)
        nsw = read_nsw(calc_dir)
        if nsw and max_step >= nsw:
            issues.append(f"NSW_REACHED({max_step}/{nsw})")

    return issues


def check_outcar(calc_dir):
    """Check OUTCAR for convergence. Returns (converged, issues)."""
    issues = []
    outcar = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.isfile(outcar):
        return False, ["NO_OUTCAR"]
    try:
        with open(outcar, 'r', errors='replace') as f:
            content = f.read()
    except Exception:
        return False, ["OUTCAR_READ_ERROR"]
    converged = 'reached required accuracy' in content
    if not converged:
        issues.append("NOT_CONVERGED")
    return converged, issues


def check_contcar(calc_dir):
    """Check if CONTCAR exists and is valid."""
    contcar = os.path.join(calc_dir, 'CONTCAR')
    if not os.path.isfile(contcar):
        return ["NO_CONTCAR"]
    if os.path.getsize(contcar) == 0:
        return ["EMPTY_CONTCAR"]
    try:
        with open(contcar) as f:
            lines = f.readlines()
        if len(lines) < 6:
            return ["INCOMPLETE_CONTCAR"]
    except Exception:
        return ["CONTCAR_READ_ERROR"]
    return []


def read_nsw(calc_dir):
    """Read NSW from INCAR."""
    incar = os.path.join(calc_dir, 'INCAR')
    if not os.path.isfile(incar):
        return None
    try:
        with open(incar) as f:
            for line in f:
                m = re.match(r'\s*NSW\s*=\s*(\d+)', line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return None


def read_incar_param(calc_dir, param):
    """Read a numeric parameter from INCAR (e.g. POTIM, IBRION)."""
    incar = os.path.join(calc_dir, 'INCAR')
    if not os.path.isfile(incar):
        return None
    try:
        with open(incar) as f:
            for line in f:
                m = re.match(rf'\s*{param}\s*=\s*([\d.Ee+-]+)', line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None


def read_lattice_constant(calc_dir):
    """Read equivalent cubic lattice constant from CONTCAR via cell volume.

    Uses vol**(1/3) instead of first-vector magnitude, which correctly
    handles non-orthogonal cells (FCC primitive, rotated, etc.).
    """
    contcar = os.path.join(calc_dir, 'CONTCAR')
    if not os.path.isfile(contcar) or os.path.getsize(contcar) == 0:
        return None
    try:
        with open(contcar) as f:
            lines = f.readlines()
        if len(lines) < 5:
            return None
        scale = float(lines[1].strip())
        a_vec = [float(x) for x in lines[2].split()]
        b_vec = [float(x) for x in lines[3].split()]
        c_vec = [float(x) for x in lines[4].split()]
        vol = abs(
            a_vec[0] * (b_vec[1]*c_vec[2] - b_vec[2]*c_vec[1]) -
            a_vec[1] * (b_vec[0]*c_vec[2] - b_vec[2]*c_vec[0]) +
            a_vec[2] * (b_vec[0]*c_vec[1] - b_vec[1]*c_vec[0])
        ) * abs(scale)**3
        return vol ** (1.0 / 3.0)
    except Exception:
        return None


# =====================================================================
# Ω_sf computation
# =====================================================================
VASP_ATOMIC_VOLUMES = {
    "Ag": 17.840, "Al": 16.602, "Au": 17.798, "Be": 8.105,
    "Ca": 42.025, "Co": 10.994, "Cr": 11.415, "Cu": 12.024,
    "Dy": 31.744, "Er": 31.063, "Fe": 11.312, "Ge": 19.243,
    "Hf": 22.068, "Ir": 14.334, "La": 37.591, "Mg": 22.909,
    "Mn": 10.855, "Mo": 15.629, "Nb": 18.370, "Ni": 10.941,
    "Os": 13.776, "Pb": 30.596, "Pd": 15.466, "Pt": 15.219,
    "Re": 14.875, "Rh": 13.761, "Ru": 14.136, "Sc": 24.869,
    "Si": 14.822, "Sn": 27.611, "Ta": 18.159, "Tb": 32.503,
    "Ti": 17.022, "V":  13.275, "W":  15.960, "Y":  33.017,
    "Zn": 15.741, "Zr": 22.721,
}


def compute_omega_sf(elA, countA, elB, countB, lattice_const):
    """Compute Ω_sf for a given pair."""
    if elA not in VASP_ATOMIC_VOLUMES or elB not in VASP_ATOMIC_VOLUMES:
        return None
    if elA == elB:
        return None
    vA = VASP_ATOMIC_VOLUMES[elA]
    vB = VASP_ATOMIC_VOLUMES[elB]
    total = countA + countB
    v_actual = lattice_const**3 / total
    v_vegard = (countA * vA + countB * vB) / total
    if v_vegard == 0:
        return None
    return (v_actual - v_vegard) / v_vegard


# =====================================================================
# Phase 1: Scan & detect
# =====================================================================
def scan_directory(sqs_dir, omega_threshold=0.5):
    """Scan all SQS directories and detect rerun cases."""
    results = []

    if not os.path.isdir(sqs_dir):
        print(f"ERROR: {sqs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(os.listdir(sqs_dir))
    total = 0
    converged = 0
    rerun_needed = 0

    for dirname in dirs:
        calc_dir = os.path.join(sqs_dir, dirname)
        if not os.path.isdir(calc_dir):
            continue
        parsed = parse_dirname(dirname)
        if parsed is None:
            continue

        elements = parsed  # list of (element, count)
        total += 1
        all_issues = []

        log_issues = check_vasp_log(calc_dir)
        all_issues.extend(log_issues)

        conv, outcar_issues = check_outcar(calc_dir)
        all_issues.extend(outcar_issues)
        if conv:
            converged += 1

        contcar_issues = check_contcar(calc_dir)
        all_issues.extend(contcar_issues)

        omega_sf = None
        elA, countA, elB, countB = None, None, None, None
        # Ω_sf check only for binary compounds
        if len(elements) == 2:
            elA, countA = elements[0]
            elB, countB = elements[1]
            if elA != elB:
                a = read_lattice_constant(calc_dir)
                if a is not None:
                    omega_sf = compute_omega_sf(elA, countA, elB, countB, a)
                    if omega_sf is not None and abs(omega_sf) > omega_threshold:
                        all_issues.append(f"OMEGA_OUTLIER({omega_sf:+.3f})")

        # Read current POTIM, IBRION, ENCUT for diagnostics
        cur_potim = read_incar_param(calc_dir, 'POTIM')
        cur_ibrion = read_incar_param(calc_dir, 'IBRION')
        cur_encut = read_incar_param(calc_dir, 'ENCUT')

        needs_rerun = len(all_issues) > 0
        if needs_rerun:
            rerun_needed += 1

        results.append({
            'dirname': dirname,
            'calc_dir': calc_dir,
            'elA': elA, 'countA': countA,
            'elB': elB, 'countB': countB,
            'converged': conv,
            'omega_sf': omega_sf,
            'issues': all_issues,
            'needs_rerun': needs_rerun,
            'potim': cur_potim,
            'ibrion': cur_ibrion,
            'encut': cur_encut,
        })

    return results, total, converged, rerun_needed


# =====================================================================
# Phase 2: Prepare & run
# =====================================================================
def backup_and_prepare(calc_dir, new_nsw=300, new_potim=None, new_ibrion=None,
                       new_encut=None, addgrid=False):
    """
    Prepare directory for rerun:
    1. Backup old outputs to .bak/
    2. Copy CONTCAR → POSCAR (restart from last geometry)
    3. Update INCAR: NSW, ISIF=7, POTIM, IBRION, ENCUT, ADDGRID
    Returns True if preparation succeeded.
    """
    # Create backup
    bak_dir = os.path.join(calc_dir, '.bak')
    os.makedirs(bak_dir, exist_ok=True)
    for fname in ['POSCAR', 'OUTCAR', 'OSZICAR', 'vasp.log',
                  'WAVECAR', 'CHGCAR', 'CHG', 'vasprun.xml']:
        src = os.path.join(calc_dir, fname)
        if os.path.isfile(src):
            dst = os.path.join(bak_dir, fname)
            shutil.copy2(src, dst)

    # CONTCAR → POSCAR (restart from last geometry)
    contcar = os.path.join(calc_dir, 'CONTCAR')
    poscar = os.path.join(calc_dir, 'POSCAR')
    if os.path.isfile(contcar) and os.path.getsize(contcar) > 100:
        shutil.copy2(contcar, poscar)
    else:
        # No valid CONTCAR — keep original POSCAR
        if not os.path.isfile(poscar):
            return False

    # Update INCAR
    incar_path = os.path.join(calc_dir, 'INCAR')
    if not os.path.isfile(incar_path):
        return False

    with open(incar_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    nsw_set = False
    isif_set = False
    ibrion_set = False
    potim_set = False
    encut_set = False
    addgrid_set = False
    for line in lines:
        # Update NSW
        if re.match(r'\s*NSW\s*=', line):
            new_lines.append(f' NSW = {new_nsw}\n')
            nsw_set = True
        # Ensure ISIF=7
        elif re.match(r'\s*ISIF\s*=', line):
            new_lines.append(' ISIF = 7\n')
            isif_set = True
        # IBRION
        elif re.match(r'\s*IBRION\s*=', line):
            if new_ibrion is not None:
                new_lines.append(f' IBRION = {new_ibrion}\n')
            else:
                new_lines.append(line)
            ibrion_set = True
        # POTIM
        elif re.match(r'\s*POTIM\s*=', line):
            if new_potim is not None:
                new_lines.append(f' POTIM = {new_potim:.6f}\n')
            else:
                new_lines.append(line)
            potim_set = True
        # ENCUT
        elif re.match(r'\s*ENCUT\s*=', line):
            if new_encut is not None:
                new_lines.append(f' ENCUT = {new_encut}\n')
            else:
                new_lines.append(line)
            encut_set = True
        # ADDGRID
        elif re.match(r'\s*ADDGRID\s*=', line):
            if addgrid:
                new_lines.append(' ADDGRID = .TRUE.\n')
            else:
                new_lines.append(line)
            addgrid_set = True
        else:
            new_lines.append(line)

    if not nsw_set:
        new_lines.append(f' NSW = {new_nsw}\n')
    if not isif_set:
        new_lines.append(' ISIF = 7\n')
    if new_potim is not None and not potim_set:
        new_lines.append(f' POTIM = {new_potim:.6f}\n')
    if new_ibrion is not None and not ibrion_set:
        new_lines.append(f' IBRION = {new_ibrion}\n')
    if new_encut is not None and not encut_set:
        new_lines.append(f' ENCUT = {new_encut}\n')
    if addgrid and not addgrid_set:
        new_lines.append(' ADDGRID = .TRUE.\n')

    with open(incar_path, 'w') as f:
        f.writelines(new_lines)

    # Remove WAVECAR/CHGCAR to start clean (avoid incompatible restart)
    for fname in ['WAVECAR', 'CHGCAR', 'CHG']:
        fpath = os.path.join(calc_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    return True


def run_vasp_job(calc_dir, vaspbin, ncore):
    """
    Run a single VASP job. Returns (dirname, success, elapsed_sec, message).

    Uses process groups to ensure MPI child processes are cleaned up on
    timeout, preventing zombie processes.
    """
    dirname = os.path.basename(calc_dir)
    log_path = os.path.join(calc_dir, 'vasp.log')

    cmd = f"mpirun -np {ncore} {vaspbin}"

    t0 = time.time()
    proc = None
    try:
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(
            cmd, shell=True, cwd=calc_dir,
            stdout=log_f, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group for clean kill
        )
        proc.wait(timeout=7200)  # 2 hour timeout per job
        log_f.close()

        elapsed = time.time() - t0
        # Quick convergence check
        outcar = os.path.join(calc_dir, 'OUTCAR')
        converged = False
        if os.path.isfile(outcar):
            with open(outcar, 'r', errors='replace') as f:
                converged = 'reached required accuracy' in f.read()
        status = "CONVERGED" if converged else "NOT_CONVERGED"
        return dirname, converged, elapsed, status
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        # Kill the entire process group (mpirun + all MPI children)
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        return dirname, False, elapsed, "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - t0
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
        return dirname, False, elapsed, f"ERROR: {e}"


def run_all_jobs(rerun_dirs, vaspbin, ncore, max_jobs):
    """Run all rerun jobs with max_jobs parallel workers."""
    total = len(rerun_dirs)
    print(f"\n{'='*70}")
    print(f"RUNNING {total} VASP JOBS  ({max_jobs} parallel × {ncore} cores each)")
    print(f"VASPBIN: {vaspbin}")
    print(f"{'='*70}\n")

    completed = 0
    succeeded = 0
    failed_dirs = []

    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        futures = {}
        for calc_dir in rerun_dirs:
            future = executor.submit(run_vasp_job, calc_dir, vaspbin, ncore)
            futures[future] = calc_dir

        for future in as_completed(futures):
            dirname, converged, elapsed, status = future.result()
            completed += 1
            if converged:
                succeeded += 1
                mark = "OK"
            else:
                failed_dirs.append(dirname)
                mark = "NG"
            print(f"  [{completed}/{total}] {mark}  {dirname:20s}  "
                  f"{elapsed:7.1f}s  {status}")

    print(f"\n{'='*70}")
    print(f"RESULTS: {succeeded}/{total} converged, "
          f"{total - succeeded} still unconverged")
    if failed_dirs:
        print(f"\nStill unconverged:")
        for d in failed_dirs:
            print(f"  {d}")
    print(f"{'='*70}")

    return succeeded, failed_dirs


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Detect & rerun unconverged SQS ISIF=7 calculations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect only (dry run)
  python detect_rerun_needed.py /path/to/BCC_SQS_ISIF7

  # Detect + rerun (8 cores × 4 jobs)
  python detect_rerun_needed.py /path/to/BCC_SQS_ISIF7 --rerun

  # Custom settings
  python detect_rerun_needed.py /path/to/BCC_SQS_ISIF7 --rerun \\
      --ncore 8 --max-jobs 4 --nsw 300 --potim 0.2

  # ZBRENT対策: POTIM調整 + IBRION=1 (quasi-Newton)
  python detect_rerun_needed.py /path/to/BCC_SQS_ISIF7 --rerun \\
      --potim 0.2 --ibrion 1

  # VASP mailing list推奨: IBRION=1 + ADDGRID + ENCUT増加
  python detect_rerun_needed.py /path/to/BCC_SQS_ISIF7 --rerun \\
      --ibrion 1 --addgrid --encut 600
""")
    parser.add_argument('sqs_dir',
                        help='Path to BCC_SQS_ISIF7 directory')
    parser.add_argument('--rerun', action='store_true',
                        help='Actually run VASP recalculations (default: detect only)')
    parser.add_argument('--ncore', type=int, default=8,
                        help='MPI cores per VASP job (default: 8)')
    parser.add_argument('--max-jobs', type=int, default=4,
                        help='Max parallel VASP jobs (default: 4)')
    parser.add_argument('--nsw', type=int, default=300,
                        help='NSW for rerun INCAR (default: 300)')
    parser.add_argument('--potim', type=float, default=None,
                        help='POTIM for rerun INCAR (default: auto-adjust '
                             'for ZBRENT cases). '
                             'Typical: 0.1-0.3 for ISIF=7')
    parser.add_argument('--ibrion', type=int, default=None, choices=[1, 2],
                        help='IBRION for rerun (1=RMM-DIIS quasi-Newton, '
                             '2=CG). Default: auto (ZBRENT→1, others→keep)')
    parser.add_argument('--encut', type=int, default=None,
                        help='ENCUT for rerun [eV] (default: auto — '
                             'increase to 600 for unconverged cases). '
                             'Higher ENCUT improves stress tensor accuracy '
                             'for ISIF=7 volume relaxation.')
    parser.add_argument('--addgrid', action='store_true',
                        help='Add ADDGRID=.TRUE. to INCAR '
                             '(improves FFT grid interpolation accuracy, '
                             'recommended for ISIF>0 stress calculations)')
    parser.add_argument('--omega-threshold', type=float, default=0.5,
                        help='|Ω_sf| outlier threshold (default: 0.5)')
    parser.add_argument('-o', '--output', default='rerun_list.txt',
                        help='Output file for rerun list (default: rerun_list.txt)')
    parser.add_argument('--all', action='store_true',
                        help='Show all directories including converged')
    args = parser.parse_args()

    # ---- Phase 1: Detect ----
    print(f"Scanning: {args.sqs_dir}")
    print(f"Ω_sf outlier threshold: |Ω_sf| > {args.omega_threshold}")
    print()

    results, total, converged, rerun_needed = scan_directory(
        args.sqs_dir, args.omega_threshold)

    # Summary
    print("=" * 70)
    print(f"SUMMARY")
    print(f"  Total directories:  {total}")
    print(f"  Converged (OUTCAR): {converged} ({100*converged/max(total,1):.1f}%)")
    print(f"  Rerun needed:       {rerun_needed} ({100*rerun_needed/max(total,1):.1f}%)")
    print("=" * 70)
    print()

    # Issue breakdown
    issue_counts = defaultdict(int)
    for r in results:
        for iss in r['issues']:
            key = re.sub(r'\(.*\)', '', iss)
            issue_counts[key] += 1

    if issue_counts:
        print("Issue breakdown:")
        for key, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {key:30s} {count:5d}")
        print()

    # POTIM distribution for problem cases
    zbrent_cases = [r for r in results
                    if any('ZBRENT' in i for i in r['issues'])]
    if zbrent_cases:
        potim_vals = [r['potim'] for r in zbrent_cases if r['potim'] is not None]
        if potim_vals:
            print(f"ZBRENT error cases ({len(zbrent_cases)}) — current POTIM:")
            from collections import Counter
            pcounts = Counter(potim_vals)
            for pval, cnt in sorted(pcounts.items()):
                print(f"  POTIM={pval:.6f}  ({cnt} cases)")
            print()

    # Detailed list
    print("RERUN NEEDED:")
    print("-" * 90)
    rerun_entries = []
    for r in results:
        if r['needs_rerun'] or args.all:
            flag = "RERUN" if r['needs_rerun'] else "OK"
            issues_str = ", ".join(r['issues']) if r['issues'] else "OK"
            omega_str = (f"Ω={r['omega_sf']:+.4f}"
                         if r['omega_sf'] is not None else "Ω=N/A")
            potim_str = (f"POTIM={r['potim']:.4f}"
                         if r['potim'] is not None else "POTIM=N/A")
            print(f"  {flag:5s}  {r['dirname']:20s}  {omega_str:14s}  "
                  f"{potim_str:14s}  {issues_str}")
            if r['needs_rerun']:
                rerun_entries.append(r)

    # Write rerun list
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(args.sqs_dir)), args.output)
    with open(output_path, 'w') as f:
        for r in rerun_entries:
            f.write(r['dirname'] + '\n')
    print(f"\nRerun list: {output_path} ({len(rerun_entries)} directories)")

    if not args.rerun:
        print("\n(Dry run — add --rerun to execute VASP recalculations)")
        return

    # ---- Phase 2: Rerun ----
    if not rerun_entries:
        print("\nNo directories need rerun. Done.")
        return

    # Check VASPBIN
    vaspbin = os.environ.get('VASPBIN')
    if not vaspbin:
        print("\nERROR: $VASPBIN is not set.", file=sys.stderr)
        print("  export VASPBIN=/path/to/vasp_std", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(vaspbin):
        print(f"\nERROR: VASPBIN={vaspbin} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"PREPARING {len(rerun_entries)} directories for rerun...")
    potim_desc = (f"POTIM → {args.potim}" if args.potim
                  else "POTIM → auto (ZBRENT: 0.2, others: keep)")
    ibrion_desc = (f"IBRION → {args.ibrion}" if args.ibrion
                   else "IBRION → auto (ZBRENT: 1, others: keep)")
    encut_desc = (f"ENCUT → {args.encut} eV" if args.encut
                  else "ENCUT → auto (unconverged: 600, others: keep)")
    addgrid_desc = "ADDGRID = .TRUE." if args.addgrid else "ADDGRID: keep"
    print(f"  CONTCAR → POSCAR, NSW → {args.nsw}, ISIF = 7")
    print(f"  {potim_desc}")
    print(f"  {ibrion_desc}")
    print(f"  {encut_desc}")
    print(f"  {addgrid_desc}")
    print(f"{'='*70}")

    # Determine POTIM/IBRION/ENCUT/ADDGRID strategy per directory
    prepared_dirs = []
    for r in rerun_entries:
        calc_dir = r['calc_dir']
        has_zbrent = any('ZBRENT' in i for i in r['issues'])

        # POTIM: user-specified > auto-adjust for ZBRENT > keep original
        if args.potim is not None:
            potim = args.potim
        elif has_zbrent:
            cur = r.get('potim')
            if cur is not None and cur < 0.1:
                potim = 0.2  # too small → increase
            elif cur is not None and cur > 0.5:
                potim = 0.2  # too large → decrease
            else:
                potim = 0.2  # default ZBRENT fix
        else:
            potim = None  # keep original

        # IBRION: user-specified > auto (ZBRENT→1) > keep original
        if args.ibrion is not None:
            ibrion = args.ibrion
        elif has_zbrent:
            ibrion = 1  # RMM-DIIS quasi-Newton: more robust near minimum
        else:
            ibrion = None  # keep original

        # ENCUT: user-specified > auto (increase to 600 if below) > keep original
        cur_encut = r.get('encut')
        if args.encut is not None:
            encut = args.encut
        elif cur_encut is None or cur_encut < 600:
            encut = 600  # increase for unconverged cases
        else:
            encut = None  # keep original (already >= 600)

        # ADDGRID: from CLI flag
        addgrid = args.addgrid

        ok = backup_and_prepare(calc_dir, new_nsw=args.nsw,
                                new_potim=potim, new_ibrion=ibrion,
                                new_encut=encut, addgrid=addgrid)
        if ok:
            prepared_dirs.append(calc_dir)
            potim_msg = f"POTIM={potim}" if potim else "POTIM=keep"
            ibrion_msg = f"IBRION={ibrion}" if ibrion else "IBRION=keep"
            encut_msg = f"ENCUT={encut}" if encut else "ENCUT=keep"
            addgrid_msg = "ADDGRID=T" if addgrid else ""
            extras = f"  {encut_msg}" + (f"  {addgrid_msg}" if addgrid_msg else "")
            print(f"  OK  {r['dirname']:20s}  {potim_msg}  {ibrion_msg}{extras}")
        else:
            print(f"  NG  {r['dirname']}  (preparation failed, skipping)")

    if not prepared_dirs:
        print("\nNo directories could be prepared. Aborting.")
        return

    # Confirm before running
    print(f"\nReady to run {len(prepared_dirs)} VASP jobs")
    print(f"  {args.max_jobs} parallel × {args.ncore} cores = "
          f"{args.max_jobs * args.ncore} total cores")
    print(f"  VASPBIN: {vaspbin}")
    resp = input("Proceed? [y/N] ").strip().lower()
    if resp != 'y':
        print("Aborted.")
        return

    # Run
    succeeded, failed = run_all_jobs(
        prepared_dirs, vaspbin, args.ncore, args.max_jobs)

    # Write remaining failures
    if failed:
        fail_path = os.path.join(
            os.path.dirname(os.path.abspath(args.sqs_dir)),
            'still_unconverged.txt')
        with open(fail_path, 'w') as f:
            for d in failed:
                f.write(d + '\n')
        print(f"\nStill-unconverged list: {fail_path}")


if __name__ == '__main__':
    main()
