#ifndef CARPETX_PLANESX_CARPETX_PARAMS_HXX
#define CARPETX_PLANESX_CARPETX_PARAMS_HXX

// PlanesX follows CarpetX's output conventions (file mode, cadence fallback,
// compression, openPMD format), but those CarpetX parameters are private, so
// they cannot be shared via param.ccl USES. Query them at runtime instead.

#include <cctk.h>
#include <cctk_Parameter.h>

#include <cassert>

namespace PlanesX {

inline CCTK_INT get_carpetx_int_param(const char *const name) {
  int type;
  const void *const ptr = CCTK_ParameterGet(name, "CarpetX", &type);
  if (!ptr)
    CCTK_VERROR("Parameter CarpetX::%s not found (renamed upstream?)", name);
  assert(type == PARAMETER_INT);
  return *static_cast<const CCTK_INT *>(ptr);
}

inline const char *get_carpetx_string_param(const char *const name) {
  int type;
  const void *const ptr = CCTK_ParameterGet(name, "CarpetX", &type);
  if (!ptr)
    CCTK_VERROR("Parameter CarpetX::%s not found (renamed upstream?)", name);
  assert(type == PARAMETER_STRING || type == PARAMETER_KEYWORD);
  return *static_cast<const char *const *>(ptr);
}

} // namespace PlanesX

#endif // #ifndef CARPETX_PLANESX_CARPETX_PARAMS_HXX
