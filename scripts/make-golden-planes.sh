#!/bin/bash
#
# Generate the golden 2D-plane reference output (.bp5 + .silo) for thorn
# TestPlanesX, in place, on a machine that can run the CarpetX executable
# (e.g. Frontier). No Python / openpmd_api is needed -- it only runs the
# simulation and copies the produced plane files into the repo, ready to
# `git add` and commit.
#
# By default every case runs as a single serial process (no srun/mpirun): the
# verifier compares golden vs fresh data point-by-point, so the reference does
# not need to match CI's MPI decomposition. This makes the script easy to run
# on a login node.
#
# Usage (from anywhere; the script finds the repo via its own location):
#
#   EXE=/path/to/Cactus/exe/cactus_sim bash scripts/make-golden-planes.sh
#
# Environment:
#   EXE       Path to the cactus executable (REQUIRED unless CACTUS_DIR is set
#             and the executable lives at $CACTUS_DIR/exe/cactus_sim).
#   CACTUS_DIR  Cactus dir; EXE defaults to $CACTUS_DIR/exe/cactus_sim.
#   MPIRUN    MPI launcher; empty by default (serial). Only used if NPROCS > 1,
#             e.g. NPROCS=2 MPIRUN="srun" for a parallel run.
#   NPROCS    Ranks to run each case on (default: 1).
#   GOLDEN_INCLUDE_SILO  If 1, also archive the (large, not-CI-compared) .silo
#             output. Default 0: only the openPMD .bp5 output is kept.
#   WORKDIR   Scratch run directory (default: a fresh mktemp dir).
#
# The reference is written to TestPlanesX/test/golden/<parfile>/ for each parfile.
# Only the openPMD .bp5 output is kept by default -- it is what the verifier
# compares in CI; the Silo .silo output is large and not numerically compared
# in CI, so it is omitted (set GOLDEN_INCLUDE_SILO=1 to keep it).

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARPETX_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
TESTDIR="$CARPETX_REPO/TestPlanesX/test"
GOLDEN="$TESTDIR/golden"

CACTUS_DIR="${CACTUS_DIR:-$CARPETX_REPO/../workspace/Cactus}"
EXE="${EXE:-$CACTUS_DIR/exe/cactus_sim}"
MPIRUN="${MPIRUN-}"
NPROCS="${NPROCS:-1}"
WORKDIR="${WORKDIR:-$(mktemp -d)}"

if [ ! -x "$EXE" ]; then
  echo "✗ cactus executable not found or not executable: $EXE" >&2
  echo "  Set EXE=/path/to/exe/cactus_sim (or CACTUS_DIR=...)." >&2
  exit 1
fi

# OpenMPI refuses to run as root unless told otherwise (harmless elsewhere).
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

echo "executable : $EXE"
echo "ranks      : $NPROCS${MPIRUN:+ via $MPIRUN}"
echo "workdir    : $WORKDIR"
echo "golden dir : $GOLDEN"
echo

mkdir -p "$GOLDEN"

make_one() {
  par_base="$1"
  par="$TESTDIR/$par_base.par"
  echo "================================================================"
  echo "Generating golden for $par_base on $NPROCS rank(s)"
  rm -rf "${WORKDIR:?}/$par_base"
  (
    cd "$WORKDIR"
    if [ -n "$MPIRUN" ] && [ "$NPROCS" -gt 1 ]; then
      "$MPIRUN" -n "$NPROCS" "$EXE" "$par"
    else
      "$EXE" "$par"
    fi
  )
  if [ ! -d "$WORKDIR/$par_base" ]; then
    echo "✗ expected output directory $WORKDIR/$par_base was not created" >&2
    exit 1
  fi
  dest="$GOLDEN/$par_base"
  rm -rf "$dest"
  mkdir -p "$dest"
  # Copy only the openPMD (.bp5) plane output -- that is what the verifier
  # numerically compares in CI. The Silo (.silo) output is large (one HDF5
  # object per AMR component) and is not numerically compared in CI (no Silo
  # Python reader there), so it is excluded by default. Set
  # GOLDEN_INCLUDE_SILO=1 to archive it anyway. The run-timing performance.yaml
  # is never golden.
  shopt -s nullglob
  for f in "$WORKDIR/$par_base/"*.it*.bp* "$WORKDIR/$par_base/"*.openpmd.visit; do
    cp -r "$f" "$dest/"
  done
  if [ "${GOLDEN_INCLUDE_SILO:-0}" = "1" ]; then
    for f in "$WORKDIR/$par_base/"*.it*.silo \
             "$WORKDIR/$par_base/"*.silo_planes.dir \
             "$WORKDIR/$par_base/"*.silo_planes.visit; do
      cp -r "$f" "$dest/"
    done
  fi
  shopt -u nullglob
  # Drop ADIOS2 run-timing profiling dumps: machine/run-specific, change on
  # every regeneration, and not read by the verifier (it uses md.idx/data/md/mmd).
  find "$dest" -name profiling.json -delete
  echo "  wrote $(find "$dest" -type f | wc -l) files ($(du -sh "$dest" | cut -f1)) to $dest"
}

make_one planes-single-level
make_one planes-amr
make_one planes-amr-midlevel
make_one planes-edge-cases
make_one planes-int-tags

echo "================================================================"
echo "Done. Review the new files, then:"
echo "    git add TestPlanesX/test/golden"
echo "    git commit -m 'TestPlanesX: add golden plane output (.bp5 + .silo)'"
