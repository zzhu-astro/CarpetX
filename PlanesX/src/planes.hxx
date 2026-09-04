#ifndef CARPETX_PLANESX_PLANES_HXX
#define CARPETX_PLANESX_PLANES_HXX

#include <cctk.h>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>

#include <array>
#include <cassert>
#include <string>
#include <vector>

namespace PlanesX {

// normal_axis: 0 -> yz, 1 -> xz, 2 -> xy. elevation is already rounded
// to frac_precision; tag is the filename/meshname fragment.
struct plane_spec_t {
  int normal_axis;
  CCTK_REAL elevation;
  std::string tag;
};

// Rounds each elevation to frac_precision digits; warns once per unique
// spec on precision overflow.
std::vector<plane_spec_t> parse_planes(const std::string &spec,
                                       int int_precision, int frac_precision);

std::string format_plane_tag(int normal_axis, CCTK_REAL rounded_elevation,
                             int int_precision, int frac_precision);

// In-plane axes (a, b) for a plane with the given normal:
// 0 -> (1, 2), 1 -> (0, 2), 2 -> (0, 1).
inline std::array<int, 2> in_plane_axes(const int normal_axis) {
  switch (normal_axis) {
  case 0:
    return {1, 2};
  case 1:
    return {0, 2};
  case 2:
    return {0, 1};
  }
  assert(0);
  return {0, 1};
}

// Returns -1 if elevation lies outside the level along normal_axis.
// VC indextype snaps to x0+i*dx; CC to x0+(i+0.5)*dx.
int snap_to_grid_index(const plane_spec_t &plane, const amrex::Geometry &geom,
                       const amrex::IndexType &indextype);

// World coordinate of the snapped slab along the normal axis: the true
// location of data written for this (plane, level, normal centering). NaN if
// the plane snaps outside the level (snap_to_grid_index < 0). Only the
// centering along the normal axis matters.
CCTK_REAL snapped_plane_coordinate(const plane_spec_t &plane,
                                   const amrex::Geometry &geom,
                                   const amrex::IndexType &indextype);

// True if the plane's elevation snaps onto at least one Cartesian
// (patch, level) of an enabled grid-function group. Purely geometric (no
// rank-local data), so all ranks agree.
bool plane_in_any_domain(const plane_spec_t &plane,
                         const std::vector<bool> &output_group);

// Collapses fab along normal_axis at slice_idx into out_buf (size
// numvars * in-plane cells, var-major). Returns empty box if out of range.
amrex::Box extract_slab(const amrex::FArrayBox &fab, int normal_axis,
                        int slice_idx, int numvars,
                        std::vector<CCTK_REAL> &out_buf);

// Single-pass restricted variant: copies the thickness-1 slab clipped to
// target's in-plane extent (target must lie within fab.box() and contain
// slice_idx along the normal) into dst, var-major and axis_a-fastest, of size
// numvars * target.length(a) * target.length(b). Used by the openPMD writer
// to stage the interior directly, without an intermediate full-ghost copy.
void extract_slab_into(const amrex::FArrayBox &fab, int normal_axis,
                       int slice_idx, int numvars, const amrex::Box &target,
                       CCTK_REAL *dst);

} // namespace PlanesX

#endif // #ifndef CARPETX_PLANESX_PLANES_HXX
