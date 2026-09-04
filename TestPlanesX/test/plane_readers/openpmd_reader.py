"""openPMD plane reader.

Reads PlanesX 2D-plane openPMD output (ADIOS2 .bp/.bp4/.bp5 or HDF5 .h5) via
openpmd_api and returns a list of `Slab`s. Interior-only data, matching the
writer's convention. Geometry-free and reusable.
"""

import glob
import os
import re

from .slab import Slab, centering_from_varname, in_plane_axes


def read_openpmd_planes(out_dir, sim):
    """Read all openPMD plane files `<sim>.<tag>.it<N>.<ext>` in out_dir.

    Returns (records, ok, msg) where records is a list of
    (tag, normal_axis, elevation, [Slab, ...]) and ok is False only if
    openpmd_api is unavailable.
    """
    try:
        import openpmd_api as io
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return [], False, "openpmd_api unavailable: %s" % exc

    # Group files by (tag, ext): <sim>.<tag>.it<NNNNNNNN>.<ext>
    groups = {}
    for ext in ("bp5", "bp", "bp4", "h5"):
        for path in glob.glob(os.path.join(out_dir, "%s.*.it*.%s" % (sim, ext))):
            m = re.match(r"^%s\.(.+)\.it(\d+)\.%s$" % (re.escape(sim), ext),
                         os.path.basename(path))
            if m:
                groups[(m.group(1), ext)] = True

    records = []
    skipped = []
    for (tag, ext) in sorted(groups):
        pattern = os.path.join(out_dir, "%s.%s.it%%08T.%s" % (sim, tag, ext))
        try:
            series = io.Series(pattern, io.Access.read_only)
        except Exception as exc:  # noqa: BLE001
            # A file that cannot be opened (e.g. an out-of-domain plane the
            # writer should not have produced) is skipped here; the caller can
            # assert separately that no such file exists.
            skipped.append(tag)
            continue
        try:
            for it_index in series.iterations:
                it = series.iterations[it_index]
                normal_axis = int(it.get_attribute("planeNormalAxis"))
                elevation = float(it.get_attribute("planeElevation"))
                axis_a, axis_b = in_plane_axes(normal_axis)

                # Queue every chunk load, then flush once for the whole file.
                pending = []
                for mesh_name in it.meshes:
                    mesh = it.meshes[mesh_name]
                    lev_m = re.search(r"_lev(\d+)", mesh_name)
                    patch_m = re.search(r"_patch(\d+)", mesh_name)
                    level = int(lev_m.group(1)) if lev_m else 0
                    patch = int(patch_m.group(1)) if patch_m else 0
                    ggo = list(mesh.grid_global_offset)   # [off_b, off_a]
                    gsp = list(mesh.grid_spacing)          # [sp_b, sp_a]
                    # True (snapped) normal coordinate recorded per mesh;
                    # absent in files written before this metadata existed.
                    try:
                        ncoord = (float(mesh.get_attribute("planeCoordinate"))
                                  if mesh.contains_attribute("planeCoordinate")
                                  else None)
                    except Exception:  # noqa: BLE001
                        ncoord = None
                    for comp_name in mesh:
                        centering = centering_from_varname(comp_name)
                        if centering is None:
                            continue
                        rc = mesh[comp_name]
                        pos = list(rc.position)            # [pos_b, pos_a]
                        shape = list(rc.shape)             # [N_b, N_a] vertices
                        for ch in rc.available_chunks():
                            off = [int(v) for v in ch.offset]   # [o_b, o_a]
                            ext_ = [int(v) for v in ch.extent]  # [c_b, c_a]
                            # The dataset extent is the vertex count for every
                            # centering; a cell-centred axis writes only ncells
                            # = N-1, leaving the high-edge index as backend
                            # fill (see plane_output.md). ADIOS2 reports only
                            # the written boxes, but HDF5 has no chunk
                            # bookkeeping and reports the full extent as one
                            # chunk -- clip the fill index off cell-centred
                            # axes (a no-op for ADIOS2 box chunks).
                            for d, axis in ((0, axis_b), (1, axis_a)):
                                if centering[axis]:
                                    valid = int(shape[d]) - 1
                                    ext_[d] = min(ext_[d],
                                                  max(0, valid - off[d]))
                            if ext_[0] == 0 or ext_[1] == 0:
                                continue
                            pending.append((comp_name, centering, level, patch,
                                            ggo, gsp, pos, off, ext_, ncoord,
                                            rc.load_chunk(off, ext_)))
                series.flush()

                slabs = []
                for (var, cent, level, patch, ggo, gsp, pos,
                     off, ext_, ncoord, arr) in pending:
                    nb, na = int(ext_[0]), int(ext_[1])
                    coords_b = [ggo[0] + (off[0] + j + pos[0]) * gsp[0]
                                for j in range(nb)]
                    coords_a = [ggo[1] + (off[1] + i + pos[1]) * gsp[1]
                                for i in range(na)]
                    slabs.append(Slab("openpmd", var, cent, normal_axis, level,
                                      patch, axis_a, axis_b, coords_a, coords_b,
                                      np.asarray(arr).reshape(nb, na),
                                      ncoord_file=ncoord))
                records.append((tag, normal_axis, elevation, slabs))
        finally:
            del series

    msg = "ok" if not skipped else "ok (%d unreadable file(s): %s)" % (
        len(skipped), ", ".join(sorted(skipped)))
    return records, True, msg
