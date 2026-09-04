#!/usr/bin/env python3
"""Verify PlanesX 2D plane output (Silo and openPMD) for thorn TestPlanesX.

Every grid function written by TestPlanesX holds the analytic field

    f(x, y, z) = x + 100*y + 10000*z

evaluated at its own centering-dependent world coordinate. This script reads the
plane files back (via the reusable `plane_readers` package) and checks, for
every value written, that it equals f at the point's reconstructed world
coordinate -- a reference-free, analytic check of the snap, slab extraction,
axis ordering and per-centering coordinates.

In-plane coordinates come from each file's own mesh metadata (so a wrong
gridSpacing/offset/position/centering is caught); the normal-axis coordinate is
obtained by independently replaying PlanesX::snap_to_grid_index from the parfile
geometry (so a wrong slice is caught). The two are sourced differently so
neither can mask a bug in the other. The writers additionally record the true
snapped coordinate in the file (openPMD per-mesh planeCoordinate; Silo
plane_ncoord_* arrays); that recorded value is cross-checked against the snap
replay, so the file metadata users rely on is itself verified. It also checks coarse-level coverage (the
interior points must tile the whole domain plane -> catches silently dropped
data) and, if --golden-dir is given and holds committed reference output for
this parfile, compares fresh-vs-golden data (decomposition-independent).

This file owns only the parfile/geometry handling and the comparison logic; all
file reading lives in plane_readers/ so it can be reused elsewhere.

Usage:
    verify_planes.py --parfile P.par --out-dir DIR [--golden-dir D]
                     [--silo-mode {auto,verify,smoke,skip,inspect}]
                     [--require-openpmd] [--require-silo]
"""

import argparse
import glob
import math
import os
import re
import sys

from plane_readers import (read_openpmd_planes, read_silo_planes, inspect_silo,
                           parse_plane_tag)

# f(x,y,z) = X_WEIGHTS . (x,y,z)
X_WEIGHTS = (1.0, 100.0, 10000.0)
AXIS_NAMES = ("x", "y", "z")

# Default comparison tolerances (real64). f is linear and the grid coordinates
# are exact dyadic rationals, so agreement is in fact bit-exact; this is margin.
DEFAULT_RTOL = 1.0e-9
DEFAULT_ATOL = 1.0e-6


def f_analytic(x, y, z):
    return X_WEIGHTS[0] * x + X_WEIGHTS[1] * y + X_WEIGHTS[2] * z


def lround_half_away(x):
    """Match C++ std::lround: round half away from zero."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


# ---------------------------------------------------------------------------
# Parfile parsing (geometry + plane specs + tag precision)
# ---------------------------------------------------------------------------

class Geometry:
    def __init__(self, par):
        self.ncells = [int(par["carpetx::ncells_%s" % a]) for a in AXIS_NAMES]
        self.xmin = [float(par["carpetx::%smin" % a]) for a in AXIS_NAMES]
        self.xmax = [float(par["carpetx::%smax" % a]) for a in AXIS_NAMES]
        self.max_num_levels = int(par.get("carpetx::max_num_levels", "1"))
        self.int_precision = int(par.get("planesx::planes_int_precision", "4"))
        self.frac_precision = int(par.get("planesx::planes_frac_precision", "3"))

    def dx(self, axis, level):
        return (self.xmax[axis] - self.xmin[axis]) / self.ncells[axis] / 2 ** level

    def ncells_at(self, axis, level):
        return self.ncells[axis] * 2 ** level

    def in_domain(self, axis, elevation):
        return self.xmin[axis] <= elevation <= self.xmax[axis]

    def snap_coord(self, axis, level, cell_centered, elevation):
        """Replay PlanesX::snap_to_grid_index; return the snapped slice's world
        coordinate, or None if the elevation is out of range."""
        x0 = self.xmin[axis]
        dx = self.dx(axis, level)
        ncells = self.ncells_at(axis, level)
        r = (elevation - x0) / dx
        if cell_centered:
            i = lround_half_away(r - 0.5)
            return None if i < 0 or i > ncells - 1 else x0 + (i + 0.5) * dx
        i = lround_half_away(r)
        return None if i < 0 or i > ncells else x0 + i * dx


def round_elevation(elev, frac_precision):
    """Replay parse_planes elevation rounding (round half away from zero)."""
    scale = 10 ** frac_precision
    sign = -1.0 if elev < 0 else 1.0
    return sign * (lround_half_away(abs(elev) * scale) / scale)


def parse_parfile(path):
    """Return a dict of lowercased 'thorn::param' / '$var' -> string value.

    Handles double-quoted (possibly multi-line) and bare values, strips '#'
    comments outside quotes, and resolves '$var' references.
    """
    with open(path) as fh:
        text = fh.read()
    lines, in_str = [], False
    for raw in text.splitlines():
        buf = []
        for ch in raw:
            if ch == '"':
                in_str = not in_str
                buf.append(ch)
            elif ch == "#" and not in_str:
                break
            else:
                buf.append(ch)
        lines.append("".join(buf))

    params = {}
    pat = re.compile(r'([\w:$]+)\s*=\s*("(?:[^"]*)"|\S+)', re.DOTALL)
    for m in pat.finditer("\n".join(lines)):
        val = m.group(2).strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        params[m.group(1).strip().lower()] = val
    for name, val in list(params.items()):       # resolve "$var"
        if val.strip().startswith("$") and val.strip().lower() in params:
            params[name] = params[val.strip().lower()]
    return params


# ---------------------------------------------------------------------------
# Analytic + coverage verification
# ---------------------------------------------------------------------------

def verify_slabs(records, geom, rtol, atol):
    """Check every value == f at its reconstructed coordinate, and that the
    coarse level is fully covered. Returns (n_checked, errors)."""
    import numpy as np
    n_checked = 0
    errors = []
    eps = 1e-9
    coverage = {}     # (tag, centering) -> set of interior (a,b) on level 0
    cover_meta = {}
    for (tag, normal_axis, elevation, slabs) in records:
        rounded = round_elevation(elevation, geom.frac_precision)
        wn = X_WEIGHTS[normal_axis]
        for s in slabs:
            ncoord = geom.snap_coord(normal_axis, s.level,
                                     bool(s.centering[normal_axis]), rounded)
            if ncoord is None:
                errors.append("%s %s L%d: produced a slab but elevation %.6g "
                              "snaps out of range" % (tag, s.var, s.level, rounded))
                continue
            # The writer records the slab's true normal coordinate in the file
            # (openPMD planeCoordinate; Silo plane_ncoord_*). Check it against
            # the independent snap replay; freshly written files must have it.
            if s.ncoord_file is None:
                errors.append("%s %s L%d: file does not record the plane "
                              "coordinate (planeCoordinate / plane_ncoord_* "
                              "missing)" % (tag, s.var, s.level))
            elif not abs(s.ncoord_file - ncoord) <= atol + rtol * abs(ncoord):
                errors.append("%s %s L%d: file-recorded plane coordinate %.10g "
                              "disagrees with snap replay %.10g"
                              % (tag, s.var, s.level, s.ncoord_file, ncoord))
            A = np.asarray(s.coords_a, dtype=float)
            B = np.asarray(s.coords_b, dtype=float)
            vals = np.asarray(s.values, dtype=float)
            expected = (wn * ncoord + X_WEIGHTS[s.axis_b] * B[:, None]
                        + X_WEIGHTS[s.axis_a] * A[None, :])
            n_checked += vals.size
            bad = np.abs(vals - expected) > (atol + rtol * np.abs(expected))
            if bad.any():
                js, isx = np.nonzero(bad)
                for k in range(min(3, len(js))):
                    j, i = int(js[k]), int(isx[k])
                    errors.append(
                        "%s %s L%d patch%d at (a=%.6g,b=%.6g,n=%.6g): got %.10g "
                        "expected %.10g" % (tag, s.var, s.level, s.patch, A[i],
                                            B[j], ncoord, vals[j, i],
                                            expected[j, i]))
                errors.append("%s %s L%d patch%d: %d of %d values differ"
                              % (tag, s.var, s.level, s.patch, int(bad.sum()),
                                 vals.size))
            # Coverage on level 0, keyed on centering so a grid function's boxes
            # (Silo names each box's quadvar separately) aggregate to the domain.
            if s.level == 0:
                ra = np.round(A[(A >= geom.xmin[s.axis_a] - eps)
                                & (A <= geom.xmax[s.axis_a] + eps)], 6)
                rb = np.round(B[(B >= geom.xmin[s.axis_b] - eps)
                                & (B <= geom.xmax[s.axis_b] + eps)], 6)
                pts = coverage.setdefault((tag, s.centering), set())
                for b in rb:
                    for a in ra:
                        pts.add((a, b))
                cover_meta[(tag, s.centering)] = (s.axis_a, s.axis_b)

    for (tag, centering), pts in coverage.items():
        axis_a, axis_b = cover_meta[(tag, centering)]
        expected_n = ((geom.ncells[axis_a] + (0 if centering[axis_a] else 1))
                      * (geom.ncells[axis_b] + (0 if centering[axis_b] else 1)))
        if len(pts) != expected_n:
            errors.append(
                "%s centering=%s L0 coverage: %d distinct interior points, "
                "expected %d (missing/extra data -- e.g. a rank's slab dropped)"
                % (tag, centering, len(pts), expected_n))
    return n_checked, errors


def find_out_of_domain_files(out_dir, sim, geom):
    """The writers must emit no file for an out-of-domain elevation. Flag any
    plane file (openPMD or Silo) whose tag decodes to one."""
    errors = []
    seen = set()
    for pat in ("%s.*.it*.h5", "%s.*.it*.bp", "%s.*.it*.bp4", "%s.*.it*.bp5",
                "%s.*.it*.silo"):
        for path in glob.glob(os.path.join(out_dir, pat % sim)):
            m = re.match(r"^%s\.(.+)\.it(\d+)\.(\w+)$" % re.escape(sim),
                         os.path.basename(path))
            if not m or m.group(1) in seen:
                continue
            seen.add(m.group(1))
            parsed = parse_plane_tag(m.group(1))
            if parsed is not None and not geom.in_domain(parsed[0], parsed[1]):
                errors.append("writer produced an output file for out-of-domain "
                              "plane '%s' (elevation %.6g on axis %d); expected "
                              "no file" % (m.group(1), parsed[1], parsed[0]))
    return errors


# ---------------------------------------------------------------------------
# Golden-reference comparison (fresh vs committed binary reference)
# ---------------------------------------------------------------------------

def slabs_to_value_map(records):
    """Map (tag, centering, level, round(a), round(b)) -> value. Keyed on
    centering (not the variable name) so it is independent of the MPI
    decomposition and of Silo's per-box variable naming."""
    import numpy as np
    m = {}
    for (tag, normal_axis, elevation, slabs) in records:
        for s in slabs:
            A = np.round(np.asarray(s.coords_a, dtype=float), 6)
            B = np.round(np.asarray(s.coords_b, dtype=float), 6)
            vals = np.asarray(s.values, dtype=float)
            for j in range(len(B)):
                row, bj = vals[j], B[j]
                for i in range(len(A)):
                    m[(tag, s.centering, s.level, A[i], bj)] = float(row[i])
    return m


def compare_to_golden(fresh_records, golden_records, rtol, atol, label):
    """Compare fresh data against golden reference data. Returns
    (n_compared, errors)."""
    fm = slabs_to_value_map(fresh_records)
    gm = slabs_to_value_map(golden_records)
    fk, gk = set(fm), set(gm)
    errors = []
    if gk - fk:
        errors.append("%s golden: %d reference point(s) missing from current "
                      "output (e.g. %s)" % (label, len(gk - fk),
                                            sorted(gk - fk)[0]))
    if fk - gk:
        errors.append("%s golden: %d current point(s) absent from reference "
                      "(e.g. %s)" % (label, len(fk - gk), sorted(fk - gk)[0]))
    nmismatch = 0
    for k in fk & gk:
        if abs(fm[k] - gm[k]) > atol + rtol * abs(gm[k]):
            if nmismatch < 3:
                errors.append("%s golden mismatch at %s: current %.10g vs "
                              "reference %.10g" % (label, k, fm[k], gm[k]))
            nmismatch += 1
    if nmismatch:
        errors.append("%s golden: %d value(s) differ from reference"
                      % (label, nmismatch))
    return len(fk & gk), errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parfile", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="directory containing the freshly produced plane output")
    ap.add_argument("--golden-dir", default=None,
                    help="directory of per-parfile golden subdirs "
                         "(<golden-dir>/<sim>/ with .bp*/.silo reference output)")
    ap.add_argument("--silo-mode", default="auto",
                    choices=["auto", "verify", "smoke", "skip", "inspect"])
    ap.add_argument("--require-openpmd", action="store_true")
    ap.add_argument("--require-silo", action="store_true")
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = ap.parse_args(argv)

    par = parse_parfile(args.parfile)
    geom = Geometry(par)
    sim = os.path.splitext(os.path.basename(args.parfile))[0]
    golden_subdir = (os.path.join(args.golden_dir, sim)
                     if args.golden_dir is not None else None)
    has_golden = golden_subdir is not None and os.path.isdir(golden_subdir)
    all_errors = []

    # Out-of-domain planes must produce no file at all.
    ood = find_out_of_domain_files(args.out_dir, sim, geom)
    if ood:
        all_errors += ood
    else:
        print("out-of-domain planes: no files produced (as expected)")

    # --- openPMD (always numeric) ------------------------------------------
    op_records, op_ok, op_msg = read_openpmd_planes(args.out_dir, sim)
    if not op_ok:
        if args.require_openpmd:
            all_errors.append("openPMD: %s" % op_msg)
        else:
            print("SKIP openPMD: %s" % op_msg)
    else:
        n, errs = verify_slabs(op_records, geom, args.rtol, args.atol)
        all_errors += errs
        print("openPMD: %d plane file groups, %d values checked, %d errors"
              % (len(op_records), n, len(errs)))
        if has_golden:
            g_records, g_ok, _ = read_openpmd_planes(golden_subdir, sim)
            ncmp, gerrs = compare_to_golden(op_records, g_records, args.rtol,
                                            args.atol, "openPMD")
            all_errors += gerrs
            print("openPMD golden: %d points compared vs reference, %d issue(s)"
                  % (ncmp, len(gerrs)))
        elif golden_subdir is not None:
            print("openPMD golden: %s absent (not enforced; generate with "
                  "scripts/make-golden-planes.sh)" % golden_subdir)

    # --- Silo --------------------------------------------------------------
    if args.silo_mode == "skip":
        print("Silo: skipped by request")
    elif args.silo_mode == "inspect":
        inspect_silo(args.out_dir, sim)
    else:
        si_records, si_mode, si_msg = read_silo_planes(args.out_dir, sim)
        print("Silo: mode=%s (%s)" % (si_mode, si_msg))
        if args.silo_mode == "verify" and si_mode != "verify":
            all_errors.append("Silo: numeric verification requested but the "
                              "reader is unavailable (%s)" % si_msg)
        elif args.require_silo and si_mode != "verify":
            all_errors.append("Silo: numeric verification required but only the "
                              "smoke check is available (%s)" % si_msg)
        if si_mode == "verify":
            n, errs = verify_slabs(si_records, geom, args.rtol, args.atol)
            all_errors += errs
            print("Silo: %d values checked, %d errors" % (n, len(errs)))
            # Silo golden is intentionally not committed (the .silo files are
            # large). When it is present the comparison mirrors the openPMD
            # path: read the golden .silo with this same reader and compare
            # data-level. Until then it is a no-op stub.
            if has_golden:
                g_records, g_mode, _ = read_silo_planes(golden_subdir, sim)
                if g_mode == "verify" and g_records:
                    ncmp, gerrs = compare_to_golden(si_records, g_records,
                                                    args.rtol, args.atol, "Silo")
                    all_errors += gerrs
                    print("Silo golden: %d points compared vs reference, "
                          "%d issue(s)" % (ncmp, len(gerrs)))
                else:
                    print("Silo golden: not committed (stub) -- skipping")

    if all_errors:
        print("\nFAILED with %d error(s):" % len(all_errors))
        for e in all_errors[:50]:
            print("  - %s" % e)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
