"""Silo plane reader.

Reads PlanesX 2D-plane Silo output (DB_HDF5 driver) for numeric verification.
The LLNL `Silo` Python module (Debian `python3-silo`) exposes only
Close/GetToc/GetVar/GetVarInfo; GetVarInfo(name) returns an object's metadata
including the HDF5 dataset paths of its components (value0 for a quadvar;
coord0/coord1 for a collinear quadmesh) and the interior node range
(min_index*/max_index*), and the arrays are read with h5py. Only interior
(non-ghost) cells are returned. Falls back to a structural smoke check when the
Silo module is unavailable. Geometry-free and reusable.
"""

import glob
import os
import re

from .slab import Slab, centering_from_varname, in_plane_axes, parse_plane_tag

# Silo centering constants (silo.h)
DB_NODECENT = 110
DB_ZONECENT = 111


def silo_meta_files(out_dir, sim):
    """Per-plane Silo metafiles (no .p<proc> suffix, not the .dir subdirs)."""
    return sorted(p for p in glob.glob(os.path.join(out_dir, "%s.*.it*.silo" % sim))
                  if ".silo_planes.dir" not in p)


def silo_data_files(out_dir, sim):
    """Per-process Silo data files inside the .silo_planes.dir subdirectories."""
    return sorted(glob.glob(os.path.join(
        out_dir, "%s.*.it*.silo_planes.dir" % sim, "*.silo")))


def read_silo_planes(out_dir, sim):
    """Return (records, mode, msg). mode is 'verify' (numeric) or 'smoke'.

    records is a list of (tag, normal_axis, elevation, [Slab, ...]).
    """
    try:
        import Silo  # noqa: F401
    except Exception:  # noqa: BLE001
        return _silo_smoke(out_dir, sim)
    try:
        records = _read_silo_module(out_dir, sim)
    except Exception as exc:  # noqa: BLE001
        recs, mode, msg = _silo_smoke(out_dir, sim)
        return recs, mode, "Silo module present but read failed (%s); %s" % (
            exc, msg)
    if not records:
        recs, mode, msg = _silo_smoke(out_dir, sim)
        return recs, mode, "Silo module present but produced no slabs; " + msg
    return records, "verify", "read via Silo Python module"


def _silo_scalar(v):
    """GetVarInfo returns each member as a length-1 list/tuple; unwrap it."""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _plane_ncoord(db, h5, cache, patch, level, cell_centred_normal):
    """True (snapped) normal coordinate recorded by the writer:
    plane_ncoord_{vertex,cell}_m<patch>[level]. Returns None when the table is
    absent (files written before this metadata existed) or unreadable."""
    import numpy as np
    kind = "cell" if cell_centred_normal else "vertex"
    key = (patch, kind)
    if key not in cache:
        name = "plane_ncoord_%s_m%04d" % (kind, patch)
        arr = None
        for read in (lambda: db.GetVar(name), lambda: h5[name][()]):
            try:
                arr = np.atleast_1d(np.asarray(read(), dtype=float))
                break
            except Exception:  # noqa: BLE001
                arr = None
        cache[key] = arr
    arr = cache[key]
    if arr is None or level >= len(arr):
        return None
    return float(arr[level])


def _read_silo_module(out_dir, sim):
    import Silo
    import h5py
    import numpy as np

    records_by_tag = {}
    for path in silo_data_files(out_dir, sim):
        base = os.path.basename(os.path.dirname(path))
        m = re.match(r"^%s\.(.+)\.it(\d+)\.silo_planes\.dir$" % re.escape(sim),
                     base)
        if not m:
            continue
        tag = m.group(1)
        parsed = parse_plane_tag(tag)
        if parsed is None:
            continue
        normal_axis, elevation = parsed
        axis_a, axis_b = in_plane_axes(normal_axis)

        db = Silo.Open(path, Silo.DB_READ)
        slabs = []
        try:
            toc = db.GetToc()
            qvar_names = list(getattr(toc, "qvar_names", []) or [])
            mesh_cache = {}
            ncoord_cache = {}
            with h5py.File(path, "r") as h5:
                def h5read(p):
                    return np.asarray(h5[p][()], dtype=float)

                for qvname in qvar_names:
                    centering = centering_from_varname(qvname)
                    if centering is None:
                        continue
                    info = db.GetVarInfo(qvname)
                    value_path = _silo_scalar(info["value0"])
                    meshid = _silo_scalar(info["meshid"])
                    silo_cent = int(_silo_scalar(info["centering"]))
                    lev_m = re.search(r"_rl(\d+)", qvname)
                    patch_m = re.search(r"_m(\d+)", qvname)
                    level = int(lev_m.group(1)) if lev_m else 0
                    patch = int(patch_m.group(1)) if patch_m else 0

                    if meshid not in mesh_cache:
                        minfo = db.GetVarInfo(meshid)
                        mesh_cache[meshid] = (
                            h5read(_silo_scalar(minfo["coord0"])),
                            h5read(_silo_scalar(minfo["coord1"])),
                            int(_silo_scalar(minfo["min_index1"])),
                            int(_silo_scalar(minfo["max_index1"])),
                            int(_silo_scalar(minfo["min_index2"])),
                            int(_silo_scalar(minfo["max_index2"])))
                    ca, cb, lo_a, hi_a, lo_b, hi_b = mesh_cache[meshid]

                    # The value buffer is axis_a-fastest with stride na;
                    # h5py's 2D shape does not match that stride for non-square
                    # arrays, so reshape (nb, na) ourselves. Then
                    # grid[j][i] == f(coord_a[i], coord_b[j]).
                    flat = np.asarray(h5[value_path][()]).reshape(-1)
                    if silo_cent == DB_ZONECENT:
                        na, nb = len(ca) - 1, len(cb) - 1
                        grid = flat.reshape(nb, na)
                        # zone centres = node-coord midpoints; interior zones
                        # lie between interior nodes [lo, hi-1].
                        coords_a = [(ca[i] + ca[i + 1]) / 2
                                    for i in range(lo_a, hi_a)]
                        coords_b = [(cb[j] + cb[j + 1]) / 2
                                    for j in range(lo_b, hi_b)]
                        sub = grid[lo_b:hi_b, lo_a:hi_a]
                    else:
                        na, nb = len(ca), len(cb)
                        grid = flat.reshape(nb, na)
                        coords_a = list(ca[lo_a:hi_a + 1])
                        coords_b = list(cb[lo_b:hi_b + 1])
                        sub = grid[lo_b:hi_b + 1, lo_a:hi_a + 1]

                    ncoord = _plane_ncoord(db, h5, ncoord_cache, patch, level,
                                           bool(centering[normal_axis]))
                    slabs.append(Slab("silo", qvname, centering, normal_axis,
                                      level, patch, axis_a, axis_b,
                                      coords_a, coords_b, np.asarray(sub),
                                      ncoord_file=ncoord))
        finally:
            db.Close()

        rec = records_by_tag.setdefault(tag, [tag, normal_axis, elevation, []])
        rec[3].extend(slabs)
    return [tuple(r) for r in records_by_tag.values()]


def _silo_smoke(out_dir, sim):
    """Structural check: every file is valid HDF5 with datasets (no values)."""
    files = silo_meta_files(out_dir, sim) + silo_data_files(out_dir, sim)
    if not files:
        return [], "smoke", "no Silo plane files found"
    try:
        import h5py
    except Exception:  # noqa: BLE001
        for path in files:
            if os.path.getsize(path) == 0:
                raise SystemExit("Silo file is empty: %s" % path)
        return [], "smoke", "%d Silo files present (size>0; no h5py)" % len(files)

    n_datasets = 0
    for path in files:
        try:
            with h5py.File(path, "r") as h5:
                def _count(name, obj):
                    nonlocal n_datasets
                    if isinstance(obj, h5py.Dataset):
                        n_datasets += 1
                h5.visititems(_count)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit("Silo file is not valid HDF5: %s (%s)" % (path, exc))
    if n_datasets == 0:
        raise SystemExit("Silo files contain no datasets (writer produced "
                         "nothing?)")
    return [], "smoke", "%d Silo files valid HDF5, %d datasets" % (
        len(files), n_datasets)


# ---------------------------------------------------------------------------
# Diagnostic: dump the on-disk layout / module API (for new containers)
# ---------------------------------------------------------------------------

def _short_attr(v):
    try:
        import numpy as np
        if isinstance(v, bytes):
            return v.decode(errors="replace")
        arr = np.asarray(v)
        return ("<%s array, size %d>" % (arr.dtype, arr.size)
                if arr.size > 12 else arr.tolist())
    except Exception:  # noqa: BLE001
        return repr(v)


def inspect_silo(out_dir, sim):
    """Dump the HDF5 layout and Silo module API for a sample plane file."""
    print("=== Silo inspection ===")
    metas = silo_meta_files(out_dir, sim)
    datas = silo_data_files(out_dir, sim)
    print("metafiles: %d, data files: %d" % (len(metas), len(datas)))
    sample = (datas or metas)
    if not sample:
        print("no Silo files found in %s" % out_dir)
        return

    try:
        import h5py
        import numpy as np
        with h5py.File(sample[0], "r") as h5:
            print("--- HDF5 tree: %s ---" % sample[0])

            def show(name, obj):
                if isinstance(obj, h5py.Group):
                    return
                line = "  [D] %s shape=%s dtype=%s" % (name, obj.shape, obj.dtype)
                if len(obj.attrs):
                    line += " attrs=%s" % {k: _short_attr(v)
                                           for k, v in obj.attrs.items()}
                print(line)
                try:
                    if obj.size and obj.size <= 16:
                        print("        values=%r"
                              % np.asarray(obj[()]).ravel().tolist())
                except Exception:  # noqa: BLE001
                    pass

            h5.visititems(show)
    except Exception as exc:  # noqa: BLE001
        print("h5py inspection failed: %s" % exc)

    try:
        import Silo
        print("--- Silo module API ---")
        db = Silo.Open(sample[0], Silo.DB_READ)
        print("  dir(db): %s" % [a for a in dir(db) if not a.startswith("_")])
        toc = db.GetToc()
        qvn = list(getattr(toc, "qvar_names", []) or [])
        qmn = list(getattr(toc, "qmesh_names", []) or [])
        print("  #qvar=%d #qmesh=%d" % (len(qvn), len(qmn)))
        if qvn:
            print("  GetVarInfo(qvar[0]) -> %r" % (db.GetVarInfo(qvn[0]),))
        if qmn:
            print("  GetVarInfo(qmesh[0]) -> %r" % (db.GetVarInfo(qmn[0]),))
        db.Close()
    except Exception as exc:  # noqa: BLE001
        print("Silo module not usable: %s" % exc)
