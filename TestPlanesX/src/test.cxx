#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

#include "loop.hxx"

using std::fabs;

namespace TestPlanesX {

// Analytic test field. It is linear in the coordinates so that it is
// represented exactly by AMR prolongation/restriction (any centering, any
// refinement level), and the three distinct decade weights make an axis swap
// or transpose in a plane writer immediately visible:
//
//   f(x,y,z) = x + 100*y + 10000*z
//
// The Python verifier recomputes f at each output point's recorded world
// coordinate and compares against the stored value. Because every grid
// function below evaluates the SAME f, the only thing that differs between
// them is the centering-dependent sampling location, so a centring bug in the
// writer (e.g. emitting vertex coordinates for a cell-centred variable) is
// caught too.
static inline CCTK_REAL f(const Loop::PointDesc &p) {
  return p.x + 100 * p.y + 10000 * p.z;
}

extern "C" void TestPlanesX_SetError(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPlanesX_SetError;
  DECLARE_CCTK_PARAMETERS;

  // Flag a cube at the domain centre for refinement (used by the AMR parfiles;
  // harmless for single-level runs since regridding is then disabled). The
  // half-size is per-level (refined_radius[L]); a finer level can nest strictly
  // inside a coarser one, so a plane can intersect coarse/middle levels but not
  // the finest. The setup is fixed to 3 levels, so clamp the index defensively.
  const int lvl = cctk_level < 3 ? cctk_level : 2;
  Loop::loop_int<1, 1, 1>(cctkGH, [&](const Loop::PointDesc &p) {
    const bool inside = fabs(p.x - refined_center_x) <= refined_radius[lvl] &&
                        fabs(p.y - refined_center_y) <= refined_radius[lvl] &&
                        fabs(p.z - refined_center_z) <= refined_radius[lvl];
    regrid_error(p.I) = inside ? 1 : 0;
  });
}

extern "C" void TestPlanesX_Set(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestPlanesX_Set;
  DECLARE_CCTK_PARAMETERS;

  // Fill everywhere (interior + ghost zones), so no SYNC is needed. f is linear
  // and CCTK_INITIAL re-runs on every level, so every point -- interior and
  // ghost -- holds f exactly regardless of prolongation accuracy.
  Loop::loop_all<0, 0, 0>(cctkGH,
                          [&](const Loop::PointDesc &p) { gf000(p.I) = f(p); });
  Loop::loop_all<0, 0, 1>(cctkGH,
                          [&](const Loop::PointDesc &p) { gf001(p.I) = f(p); });
  Loop::loop_all<0, 1, 0>(cctkGH,
                          [&](const Loop::PointDesc &p) { gf010(p.I) = f(p); });
  Loop::loop_all<0, 1, 1>(cctkGH,
                          [&](const Loop::PointDesc &p) { gf011(p.I) = f(p); });
  Loop::loop_all<1, 0, 0>(cctkGH,
                          [&](const Loop::PointDesc &p) { gf100(p.I) = f(p); });
  Loop::loop_all<1, 0, 1>(cctkGH,
                          [&](const Loop::PointDesc &p) { gf101(p.I) = f(p); });
  Loop::loop_all<1, 1, 0>(cctkGH,
                          [&](const Loop::PointDesc &p) { gf110(p.I) = f(p); });
  Loop::loop_all<1, 1, 1>(cctkGH,
                          [&](const Loop::PointDesc &p) { gf111(p.I) = f(p); });
}

} // namespace TestPlanesX
