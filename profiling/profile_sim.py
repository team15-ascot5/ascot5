"""Profile only the ``ascot_solve_distribution`` C call with gprofng.

The script re-executes itself under ``gprofng collect app -y SIGUSR1``.  gprofng
starts paused, so input generation, input staging, and ``Ascot.simulate()``
setup are not sampled.  A wrapper around ``ascot_solve_distribution`` resumes
collection immediately before the C call and pauses it again when the call
returns.

Usage::

    python profiling/profile_sim.py

The gprofng experiment is written to ``profiling/out/simulate.er``.

Environment variables:
    OMP_NUM_THREADS
        Passed through to the profiled process.
    PYTHON
        Python executable used for the profiled process. Defaults to the
        interpreter running this script.
    GPROFNG
        gprofng executable. Defaults to ``gprofng``.
"""
import os
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKDIR = HERE / "out"
EXPERIMENT = WORKDIR / "simulate.er"
PROFILE_SIGNAL = "SIGUSR1"
PROFILE_ENV = "ASCOT_PROFILE_SIGNAL"

def _launch_under_gprofng():
    """Re-execute this script in a gprofng-owned process."""
    WORKDIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env[PROFILE_ENV] = PROFILE_SIGNAL
    env.setdefault("MPLBACKEND", "Agg")

    subprocess.run([
        os.environ.get("GPROFNG", "gprofng"),
        "collect",
        "app",
        "-p",
        "hi",
        "-y",
        PROFILE_SIGNAL,
        "-O",
        str(EXPERIMENT),
        os.environ.get("PYTHON", sys.executable),
        str(Path(__file__).resolve()),
    ], check=True, cwd=str(HERE), env=env, stdout=None)

    print("Profiling finished, check results with:")
    print(f"  gprofng display text -calltree {EXPERIMENT}")

def _run_profiled():
    """Run main.py with collection enabled only around the solver call."""
    sys.path.insert(0, str(HERE.parent))

    import signal

    import a5py.engine.solve as solve

    toggle_signal = getattr(signal, os.environ[PROFILE_ENV])
    solve_distribution = solve.LIBASCOT.ascot_solve_distribution

    def _profiled_solve_distribution(*args):
        os.kill(os.getpid(), toggle_signal)  # resume collection
        try:
            return solve_distribution(*args)
        finally:
            os.kill(os.getpid(), toggle_signal)  # pause collection

    solve.LIBASCOT.ascot_solve_distribution = _profiled_solve_distribution

    # Run the user's script.  Its setup runs while gprofng is paused, and the
    # wrapper above enables collection only for the solver call.
    runpy.run_path(str(HERE / "main.py"), run_name="__main__")

if __name__ == "__main__":
    if PROFILE_ENV in os.environ:
        _run_profiled()
    else:
        _launch_under_gprofng()
