#ifndef CARPETX_PLANESX_SILO_PLANES_HXX
#define CARPETX_PLANESX_SILO_PLANES_HXX

#include "planes.hxx"

#include <cctk.h>

#ifdef HAVE_CAPABILITY_Silo

#include <string>
#include <vector>

namespace PlanesX {

void OutputSiloPlanes(const cGH *cctkGH, const std::vector<bool> &output_group,
                      const std::vector<plane_spec_t> &planes,
                      const std::string &output_dir,
                      const std::string &output_file);

} // namespace PlanesX

#endif

#endif // #ifndef CARPETX_PLANESX_SILO_PLANES_HXX
