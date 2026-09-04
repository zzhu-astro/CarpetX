"""Reusable readers for PlanesX 2D-plane output.

Each reader turns one writer's plane files into a uniform list of `Slab`s and is
independent of the test geometry and of the other writer:

    from plane_readers import read_openpmd_planes, read_silo_planes, Slab

    records, ok, msg = read_openpmd_planes(out_dir, sim)      # openPMD .bp*/.h5
    records, mode, msg = read_silo_planes(out_dir, sim)       # Silo .silo

`records` is a list of (tag, normal_axis, elevation, [Slab, ...]). See slab.py
for the Slab layout and the name/tag decoders.
"""

from .slab import (Slab, centering_from_varname, in_plane_axes,
                   parse_plane_tag)
from .openpmd_reader import read_openpmd_planes
from .silo_reader import (read_silo_planes, inspect_silo,
                          silo_meta_files, silo_data_files,
                          DB_NODECENT, DB_ZONECENT)

__all__ = [
    "Slab", "centering_from_varname", "in_plane_axes", "parse_plane_tag",
    "read_openpmd_planes", "read_silo_planes", "inspect_silo",
    "silo_meta_files", "silo_data_files", "DB_NODECENT", "DB_ZONECENT",
]
