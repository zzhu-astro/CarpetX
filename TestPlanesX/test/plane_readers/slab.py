"""Common data record and name decoders shared by the plane readers.

A *plane reader* turns the 2D-plane output of one writer (openPMD or Silo) into
a uniform list of `Slab` objects, with no dependency on the test geometry or on
the other writer -- so the readers are reusable on their own.
"""

import re


class Slab:
    """One 2D slab of a single variable at one (patch, level).

    values is a 2D array indexed [b][a] (axis_b slow, axis_a fast); coords_a and
    coords_b are the 1D world coordinates along axis_a / axis_b, so
    values[j][i] is the field at (coords_a[i], coords_b[j]) on the slice.

    ncoord_file is the slab's true world coordinate along the normal axis *as
    recorded in the file* (openPMD per-mesh planeCoordinate attribute; Silo
    plane_ncoord_* arrays), or None for files written before that metadata
    existed.
    """

    __slots__ = ("fmt", "var", "centering", "normal_axis", "level", "patch",
                 "axis_a", "axis_b", "coords_a", "coords_b", "values",
                 "ncoord_file")

    def __init__(self, fmt, var, centering, normal_axis, level, patch,
                 axis_a, axis_b, coords_a, coords_b, values, ncoord_file=None):
        self.fmt = fmt                  # "openpmd" or "silo"
        self.var = var                  # variable name as stored
        self.centering = centering      # (cx, cy, cz), 1 == cell-centred
        self.normal_axis = normal_axis  # 0/1/2 (yz/xz/xy)
        self.level = level
        self.patch = patch
        self.axis_a = axis_a            # faster in-plane axis (0/1/2)
        self.axis_b = axis_b            # slower in-plane axis (0/1/2)
        self.coords_a = coords_a        # 1D world coords along axis_a (len na)
        self.coords_b = coords_b        # 1D world coords along axis_b (len nb)
        self.values = values            # 2D [b][a]
        self.ncoord_file = ncoord_file  # recorded normal coord, or None


def in_plane_axes(normal_axis):
    """(axis_a, axis_b) for a plane with the given normal, matching PlanesX's
    in_plane_axes(): yz->(1,2), xz->(0,2), xy->(0,1)."""
    return ([1, 2], [0, 2], [0, 1])[normal_axis]


def centering_from_varname(var):
    """Extract (cx, cy, cz) from a name containing 'gf<abc>' (1 == cell)."""
    m = re.search(r"gf([012])([012])([012])", var.lower())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def parse_plane_tag(tag):
    """Inverse of PlanesX::format_plane_tag.

    'xy_z_pos0006p000' -> (normal_axis=2, elevation=6.0). Returns None if the
    tag does not look like a plane tag.
    """
    parts = tag.split("_")
    if len(parts) < 3:
        return None
    normal = {"yz": 0, "xz": 1, "xy": 2}.get(parts[0])
    if normal is None:
        return None
    mag = parts[2]
    sign = -1.0 if mag.startswith("neg") else 1.0
    mag = mag[3:]
    if "p" in mag:
        ip, fp = mag.split("p", 1)
        elev = float(ip) + (float(fp) / (10 ** len(fp)) if fp else 0.0)
    else:
        elev = float(mag)
    return normal, sign * elev
