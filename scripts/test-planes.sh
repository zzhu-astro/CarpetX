#!/bin/bash
#
# Run the TestPlanesX parfiles with the freshly built executable and verify the
# 2D plane output (Silo + openPMD) analytically. Intended to run after a build,
# both in CI (.github/workflows/ci.yml) and locally.
#
# Environment:
#   CACTUS_DIR   Cactus source/run directory (default: $PWD/../workspace/Cactus)
#   EXE          Path to the cactus executable (default: $CACTUS_DIR/exe/cactus_sim)
#   CARPETX_DIR  CarpetX repo inside Cactus (default: $CACTUS_DIR/repos/CarpetX)
#   MPIRUN       MPI launcher (default: mpirun); set to empty to run serially.

set -eu

CACTUS_DIR="${CACTUS_DIR:-$PWD/../workspace/Cactus}"
EXE="${EXE:-$CACTUS_DIR/exe/cactus_sim}"
CARPETX_DIR="${CARPETX_DIR:-$CACTUS_DIR/repos/CarpetX}"
MPIRUN="${MPIRUN-mpiexec}"

TESTDIR="$CARPETX_DIR/TestPlanesX/test"
GOLDEN="$TESTDIR/golden"
WORKDIR="${WORKDIR:-$CACTUS_DIR/test-planes-run}"

# OpenMPI refuses to run as root unless told otherwise (containers run as root).
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# Allow more MPI ranks than cores (the interior-mode regression test runs on 4
# ranks; CI runners may have fewer cores). Ignored by non-OpenMPI launchers;
# the PRTE variable covers OpenMPI 5.
export OMPI_MCA_rmaps_base_oversubscribe=true
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:${LD_LIBRARY_PATH:-}"

if [ ! -x "$EXE" ]; then
  echo "✗ executable not found or not executable: $EXE" >&2
  exit 1
fi

# Ensure the Python readers are present. openpmd_api reads the ADIOS2/BP5 plane
# files; this needs network access (fine on GitHub runners, not on HPC login
# nodes). Errors are shown (not hidden), and PEP 668 "externally-managed"
# environments are handled with --break-system-packages.
echo "python: $(command -v python3)"; python3 --version || true
python3 -m pip --version 2>/dev/null || echo "  (no pip module)"

ensure_py() {
  mod="$1"; pkg="$2"
  if python3 -c "import $mod" 2>/dev/null; then
    echo "  $mod: already importable"
    return 0
  fi
  echo "  installing $pkg ..."
  python3 -m pip install "$pkg" \
    || python3 -m pip install --break-system-packages "$pkg" \
    || python3 -m pip install --user --break-system-packages "$pkg" \
    || true
  python3 -c "import $mod" 2>/dev/null
}

ensure_py openpmd_api openpmd-api && HAVE_OPENPMD=1 || HAVE_OPENPMD=0
ensure_py h5py h5py || true

# The Silo Python module is not on PyPI; it ships as the apt package
# python3-silo (the verifier reads quadvar/quadmesh component paths via
# GetVarInfo and the arrays via h5py).
if ! python3 -c "import Silo" 2>/dev/null; then
  echo "  installing python3-silo (apt) ..."
  if command -v apt-get >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y python3-silo) 2>&1 | tail -2 || true
  fi
fi
HAVE_SILO=0; python3 -c "import Silo" 2>/dev/null && HAVE_SILO=1

if [ "$HAVE_OPENPMD" -ne 1 ]; then
  echo "✗ openpmd_api is required for rigorous plane verification but could" >&2
  echo "  not be imported or installed (see pip output above)." >&2
  exit 1
fi

# Enforce numeric Silo verification when the Silo module is available
# (it is, via apt, in the CI container); otherwise fall back to the smoke
# check so the suite still runs in environments without it.
if [ "$HAVE_SILO" -eq 1 ]; then
  SILO_ARGS="--silo-mode verify --require-silo"
else
  echo "  (python3-silo unavailable -- Silo will use the structural smoke check)"
  SILO_ARGS="--silo-mode auto"
fi

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# parfile : nprocs : extra verify args (e.g. tolerances)
run_and_verify() {
  par_base="$1"
  nprocs="$2"
  extra_args="${3:-}"
  par="$TESTDIR/$par_base.par"
  echo "================================================================"
  echo "Running $par_base on $nprocs proc(s)"
  rm -rf "$par_base"
  # timeout: a deadlocked run (e.g. an MPI-collective mismatch) must fail the
  # job quickly instead of pinning the runner until the 6 h workflow timeout.
  if [ -n "$MPIRUN" ] && [ "$nprocs" -gt 1 ]; then
    timeout 900 "$MPIRUN" -np "$nprocs" "$EXE" "$par"
  else
    timeout 900 "$EXE" "$par"
  fi
  echo "Verifying $par_base"
  # shellcheck disable=SC2086  # SILO_ARGS/extra_args are intentionally word-split
  python3 "$TESTDIR/verify_planes.py" \
    --parfile "$par" \
    --out-dir "$par_base" \
    --golden-dir "$GOLDEN" \
    --require-openpmd \
    $SILO_ARGS $extra_args
}

# Run on >1 rank where it matters: the single-level and AMR cases exercise the
# Silo MPI gather / multi-file metafile and the openPMD per-rank collective
# chunks (the grids are forced into several boxes via max_grid_size). The
# edge-case run only needs to exercise parsing/snapping, so it runs serially.
run_and_verify planes-single-level 2
run_and_verify planes-amr 2
run_and_verify planes-amr-midlevel 2
run_and_verify planes-edge-cases 1
run_and_verify planes-int-tags 1
# Silo-only size options (interior-only + single precision): float32 data
# needs relaxed tolerances (f reaches ~1.8e5 -> ~1e-2 absolute rounding).
run_and_verify planes-options 2 "--rtol 1e-6 --atol 0.05"
# Interior-only Silo slabs on 3 AMR levels and 4 ranks: the regression test for
# the "silo_planes_ghosts = no unreadable in VisIt" bug -- with single-owner
# membership some IO ranks emit no slab for a plane, which used to leave leaf
# files VisIt rejects. 4 ranks (oversubscribed if needed) make that situation
# near-certain; the VisIt gate checks the produced files block-by-block.
run_and_verify planes-amr-noghosts 4 "--rtol 1e-6 --atol 0.05"

echo "================================================================"
echo "✓ plane verification passed"
