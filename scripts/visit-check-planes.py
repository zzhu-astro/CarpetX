"""Open every PlanesX plane database under a directory with VisIt and plot it.

Drives the running VisIt over the plane output produced by the TestPlanesX
parfiles (e.g. the CI "plane-output" artifact, or a local test-planes-run/):

  visit -nowin -cli -s scripts/visit-check-planes.py <output-root> [<imgdir>]

For every Silo metafile index (*.silo_planes.visit), EVERY leaf .silo file in
every plane directory, and every openPMD HDF5 plane file (*.it*.h5), it opens
the database, makes a Pseudocolor plot of scalars (all of them for metafiles,
a capped sample for leaf files), draws it, and queries NumZones/MinMax over
the ACTUAL plotted data -- forcing VisIt through metadata, mesh construction
(including the AMR mrgtree), and data reads. A draw whose actual dataset has
zero zones or a non-finite/empty MinMax is a FAILURE: that is exactly the
"all-white plot with a plausible colorbar" mode (the colorbar limits come
from metafile extents metadata, never touching leaf data). If <imgdir> is
given, one PNG per metafile is saved there.

Exit code 0 iff every database opened and every attempted plot drew real
data. NOTE: VisIt's CLI launcher mangles exit codes (250 observed on
success); gate on the final "checked N databases, 0 failure(s)" line instead.
"""

import glob
import math
import os
import sys

# Leaf files have one quadvar per (variable, box): plotting two of them is
# enough to force actual data reads; opening the file and counting scalars is
# what catches mesh-less ("invalid") leaf files. Metafiles are plotted fully.
LEAF_SCALAR_CAP = 2

argv = [a for a in sys.argv[1:] if not a.endswith(".py")]
root = os.path.abspath(argv[0]) if argv else os.getcwd()
imgdir = os.path.abspath(argv[1]) if len(argv) > 1 else None
if imgdir and not os.path.isdir(imgdir):
    os.makedirs(imgdir)

failures = []
nchecked = 0


def open_db(db):
    """OpenDatabase, trying the explicit OpenPMD plugin for .h5 files (the
    reader does not always claim the generic .h5 extension)."""
    if OpenDatabase(db):
        return True
    if db.endswith(".h5"):
        for plugin in ("OpenPMD_1.0", "openPMD_1.0", "OpenPMD"):
            try:
                if OpenDatabase(db, 0, plugin):
                    return True
            except Exception:  # noqa: BLE001
                pass
    return False


def scalarize(v):
    """GetQueryOutputValue returns a number, or a tuple of numbers for
    queries with several outputs (e.g. NumZones on a mesh with ghost zones
    reports total and ghost counts). Reduce to the leading number."""
    if hasattr(v, "__iter__"):
        seq = list(v)
        return float(seq[0]) if seq else 0.0
    return float(v)


def draw_has_data():
    """True iff the current (drawn) plot contains real data: a nonzero zone
    count and a finite MinMax. An all-white draw (blocks missing / unreadable
    / empty) has no zones, or MinMax fails or returns non-finite values."""
    try:
        Query("NumZones", use_actual_data=1)  # noqa: F821
        nzones = scalarize(GetQueryOutputValue())  # noqa: F821
    except Exception:  # noqa: BLE001
        return False, "NumZones query failed"
    if not nzones or nzones <= 0:
        return False, "0 zones in actual data"
    try:
        Query("MinMax", use_actual_data=1)  # noqa: F821
        mm = GetQueryOutputValue()  # noqa: F821
    except Exception:  # noqa: BLE001
        return False, "MinMax query failed"
    vals = list(mm) if hasattr(mm, "__iter__") else [mm]
    if not vals or not all(math.isfinite(float(v)) for v in vals):
        return False, "non-finite MinMax %r" % (mm,)
    return True, "zones=%d minmax=%s" % (int(nzones), vals)


def check_db(db, cap, save_image):
    global nchecked
    nchecked += 1
    rel = os.path.relpath(db, root)
    try:
        if not open_db(db):
            failures.append(rel)
            print("FAIL %-72s OpenDatabase failed" % rel)
            return
        md = GetMetaData(db)  # noqa: F821 (VisIt CLI builtin)
        scalars = [md.GetScalars(i).name for i in range(md.GetNumScalars())]
        plot = scalars[:cap] if cap else scalars
        ndrawn = 0
        why = "no scalars"
        for s in plot:
            DeleteAllPlots()  # noqa: F821
            if AddPlot("Pseudocolor", s) and DrawPlots():  # noqa: F821
                ok_data, why = draw_has_data()
                if ok_data:
                    ndrawn += 1
                else:
                    print("  no-data %-60s %s" % (s, why))
        if save_image and imgdir and ndrawn:
            sw = SaveWindowAttributes()  # noqa: F821
            sw.outputToCurrentDirectory = 0
            sw.outputDirectory = imgdir
            sw.fileName = rel.replace(os.sep, "_").replace(".", "_")
            sw.format = sw.PNG
            SetSaveWindowAttributes(sw)  # noqa: F821
            SaveWindow()  # noqa: F821
        DeleteAllPlots()  # noqa: F821
        CloseDatabase(db)  # noqa: F821
        ok = ndrawn == len(plot) and ndrawn > 0
        print("%s %-72s scalars=%d plotted=%d" % (
            "OK  " if ok else "FAIL", rel, len(scalars), ndrawn))
        if not ok:
            failures.append(rel)
    except Exception as exc:  # noqa: BLE001
        failures.append(rel)
        print("FAIL %-72s %s" % (rel, exc))
        try:  # don't leak the database into the rest of the sweep
            DeleteAllPlots()  # noqa: F821
            CloseDatabase(db)  # noqa: F821
        except Exception:  # noqa: BLE001
            pass


# Silo metafile time-series indexes: the full multimesh/multivar (+ mrgtree
# AMR shadowing) path -- what a user actually opens. Every scalar is plotted.
for db in sorted(glob.glob(os.path.join(root, "*", "*.silo_planes.visit"))):
    check_db(db, cap=0, save_image=True)

# EVERY leaf .silo file: each must open standalone and contain plottable
# data. Interior-mode output once left mesh-less leaf files on ranks that
# owned no slab for a plane; VisIt rejects such a file at open time, so
# checking only the first leaf (or none) misses it.
for d in sorted(glob.glob(os.path.join(root, "*", "*.silo_planes.dir"))):
    for leaf in sorted(glob.glob(os.path.join(d, "*.silo"))):
        check_db(leaf, cap=LEAF_SCALAR_CAP, save_image=False)

# openPMD HDF5 plane files (the default .bp5 is not readable by VisIt's
# HDF5-only openPMD reader; planes-options writes .h5).
for db in sorted(glob.glob(os.path.join(root, "*", "*.it*.h5"))):
    check_db(db, cap=0, save_image=True)

print("checked %d databases, %d failure(s)" % (nchecked, len(failures)))
for f in failures:
    print("  FAILED: %s" % f)
sys.exit(1 if (failures or nchecked == 0) else 0)
