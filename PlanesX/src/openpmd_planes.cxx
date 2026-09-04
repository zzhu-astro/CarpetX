#include "openpmd_planes.hxx"

#include "carpetx_params.hxx"
#include "planes.hxx"

#include "driver.hxx"
#include "timer.hxx"

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Parameters.h>
#include <util_Network.h>

#ifdef HAVE_CAPABILITY_openPMD_api

#include <openPMD/openPMD.hpp>

#ifdef HAVE_CAPABILITY_ADIOS2
#include <adios2.h>
#endif

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if !openPMD_HAVE_MPI
#error                                                                         \
    "PlanesX requires openPMD_api with MPI support. Please use -DopenPMD_USE_MPI=ON when building openPMD."
#endif

namespace PlanesX {
using namespace CarpetX;

namespace {

constexpr bool io_verbose = true;

// The helpers below mirror CarpetX's io_openpmd.cxx so that PlanesX plane
// files follow the exact same conventions (format, ADIOS2 options, iteration
// encoding, units, mesh/component naming) as CarpetX's 3D openPMD output,
// without CarpetX having to expose its internals.

openPMD::Format get_format() {
  const char *const openpmd_format = get_carpetx_string_param("openpmd_format");
  if (CCTK_EQUALS(openpmd_format, "HDF5"))
    return openPMD::Format::HDF5;
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
#else
  if (CCTK_EQUALS(openpmd_format, "ADIOS1"))
    return openPMD::Format::ADIOS1;
#endif
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_auto"))
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
    return openPMD::Format::ADIOS2_BP5;
#else
    return openPMD::Format::ADIOS2;
#endif
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP"))
    return openPMD::Format::ADIOS2_BP;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP4"))
    return openPMD::Format::ADIOS2_BP4;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP5"))
    return openPMD::Format::ADIOS2_BP5;
#else
  if (CCTK_EQUALS(openpmd_format, "ADIOS2"))
    return openPMD::Format::ADIOS2;
#endif
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_SST"))
    return openPMD::Format::ADIOS2_SST;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_SSC"))
    return openPMD::Format::ADIOS2_SSC;
  if (CCTK_EQUALS(openpmd_format, "JSON"))
    return openPMD::Format::JSON;
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
  if (CCTK_EQUALS(openpmd_format, "TOML"))
    return openPMD::Format::TOML;
#endif
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
  if (CCTK_EQUALS(openpmd_format, "GENERIC"))
    return openPMD::Format::GENERIC;
#endif
  CCTK_VERROR("The openPMD format \"%s\" is not supported in version %d.%d.%d "
              "of the openPMD_api library",
              openpmd_format, OPENPMDAPI_VERSION_MAJOR,
              OPENPMDAPI_VERSION_MINOR, OPENPMDAPI_VERSION_PATCH);
}

// - fileBased: One file per iteration. Needs templated file name to encode
//   iteration number.
// - groupBased: Multiple iterations per file
// - variableBased: Multiple iterations stored per variable. Needs special
//    support in the backend.
constexpr openPMD::IterationEncoding iterationEncoding =
    openPMD::IterationEncoding::fileBased;

#ifdef ADIOS2_HAVE_BLOSC2
const std::string options = R"EOS(
  {
    "adios2": {
      "dataset": {
        "operators": [
          {
            "type": "blosc",
            "parameters": {
              "clevel": "9",
              "doshuffle": "BLOSC_SHUFFLE"
            }
          }
        ]
      }
    }
  }
)EOS";
#else
const std::string options = R"EOS(
  {
    "adios2": {
      "dataset": {
        "operators": [
        ]
      }
    }
  }
)EOS";
#endif

struct Const {
  // From: CODATA Internationally recommended 2018 values of the
  // Fundamental Physical Constants
  static constexpr CCTK_REAL c = 299792458;         // m s⁻¹
  static constexpr CCTK_REAL G = 6.67430e-11;       // m³ kg⁻¹ s⁻²
  static constexpr CCTK_REAL M_solar = 1.98847e+30; // kg
};

struct Unit {
  // We use c = G = 1, and M_solar as mass unit.
  static constexpr CCTK_REAL velocity = Const::c;   // m s⁻¹
  static constexpr CCTK_REAL mass = Const::M_solar; // kg
  static constexpr CCTK_REAL length =
      Const::G * mass / (Const::c * Const::c);         // m
  static constexpr CCTK_REAL time = length / velocity; // s
};

// Allowed characters are only [A-Za-z_]
std::string make_meshname(const int gi, const int patch, const int level,
                          const int tl = 0) {
  std::string groupname = CCTK_FullGroupName(gi);
  static const std::regex colons_re("::");
  groupname = std::regex_replace(groupname, colons_re, "_");
  for (auto &ch : groupname)
    ch = std::tolower(ch);
  std::ostringstream buf;
  buf << groupname;
  if (patch != -1)
    buf << "_patch" << std::setw(2) << std::setfill('0') << patch;
  if (level != -1)
    // The suffix should be `_lvl<N>`. No `setfill`?
    buf << "_lev" << std::setw(2) << std::setfill('0') << level;
  if (tl > 0)
    buf << "_tl" << std::setw(2) << std::setfill('0') << tl;
  return buf.str();
}

// Allowed characters are only [A-Za-z_]
std::string make_componentname(const int gi, const int vi) {
  const int v0 = CCTK_FirstVarIndexI(gi);
  std::string varname = CCTK_FullVarName(v0 + vi);
  static const std::regex colons_re("::");
  varname = std::regex_replace(varname, colons_re, "_");
  for (auto &ch : varname)
    ch = std::tolower(ch);
  return varname;
}

} // namespace

void OutputOpenPMDPlanes(const cGH *const cctkGH,
                         const std::vector<bool> &output_group,
                         const std::vector<plane_spec_t> &planes,
                         const std::string &output_dir,
                         const std::string &output_file) {
  const int cctk_iteration = cctkGH->cctk_iteration;
  const CCTK_REAL cctk_time = cctkGH->cctk_time;
  const CCTK_REAL cctk_delta_time = cctkGH->cctk_delta_time;

  using std::setfill, std::setw;

  static Timer timer("OutputOpenPMDPlanes");
  Interval interval(timer);

  if (planes.empty())
    return;
  if (std::count(output_group.begin(), output_group.end(), true) == 0)
    return;

  const openPMD::Format format = get_format();

  static std::set<int> warned_noncart_patches;
  static std::set<std::string> warned_outside_domain;

  {
    const int mode = 0755;
    static std::once_flag create_directory;
    std::call_once(create_directory, [&]() {
      const int ierr = CCTK_CreateDirectory(mode, output_dir.c_str());
      assert(ierr >= 0);
    });
  }

  const int myproc = CCTK_MyProc(cctkGH);
  const int ioproc = 0;

  // Serializing the full parameter table is expensive; do it once per call
  // (parameters cannot change within one invocation), not per plane. Every
  // rank needs it: see the collective-attribute note below.
  std::string all_parameters;
  {
    char *const data = IOUtil_GetAllParameters(cctkGH, 1 /*all*/);
    all_parameters = data;
    std::free(data);
  }

  // Series/iteration metadata must be IDENTICAL on every rank: ADIOS2
  // tolerates rank-local attributes, but the HDF5 backend performs collective
  // metadata operations, so rank-divergent values (or attributes set on one
  // rank only) deadlock the collective flush. Broadcast rank 0's author and
  // hostname so multi-node runs agree.
  char author[1000] = {0};
  {
    char const *const user = getenv("USER");
    if (user)
      std::snprintf(author, sizeof author, "%s", user);
    MPI_Bcast(author, sizeof author, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
  char hostname[1000] = {0};
  {
    Util_GetHostName(hostname, sizeof hostname);
    MPI_Bcast(hostname, sizeof hostname, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  for (const auto &plane : planes) {
    const auto in_axes = in_plane_axes(plane.normal_axis);
    const int axis_a = in_axes[0], axis_b = in_axes[1];

    // Skip the plane entirely -- writing no file at all -- if its elevation
    // lies outside every Cartesian (patch, level) along the normal axis. This
    // is a purely geometric decision (no rank-local data), so all ranks agree
    // and the collective Series creation below stays consistent. It also avoids
    // emitting a file whose per-level chunkInfo attributes would all be empty.
    if (!plane_in_any_domain(plane, output_group)) {
      if (warned_outside_domain.insert(plane.tag).second)
        CCTK_VWARN(CCTK_WARN_ALERT,
                   "OutputOpenPMDPlanes: plane %s lies outside all Cartesian "
                   "(patch, level) extents; writing no file",
                   plane.tag.c_str());
      continue;
    }

    std::ostringstream fnbuf;
    fnbuf << output_dir << "/" << output_file << "." << plane.tag << ".it%08T"
          << openPMD::suffix(format);
    const std::string plane_filename = fnbuf.str();

    openPMD::Series plane_series(plane_filename, openPMD::Access::CREATE,
                                 MPI_COMM_WORLD, options);
    plane_series.setIterationEncoding(iterationEncoding);
    if (author[0])
      plane_series.setAuthor(author);
    plane_series.setMachine(hostname);

    openPMD::WriteIterations plane_write_iters = plane_series.writeIterations();
    openPMD::Iteration write_iter = plane_write_iters[cctk_iteration];
    write_iter.setTime(cctk_time);
    write_iter.setDt(cctk_delta_time);
    write_iter.setTimeUnitSI(Unit::time);

    // Iteration attributes are set on EVERY rank with identical values (all
    // inputs below are replicated): rank-0-only attributes deadlock the HDF5
    // backend's collective metadata writes (ADIOS2 happened to tolerate it).
    {
      write_iter.setAttribute("AllParameters", all_parameters);

      const int npatches = ghext->patchdata.size();
      write_iter.setAttribute<std::int64_t>("numDims", 2);
      write_iter.setAttribute<std::int64_t>("numPatches", npatches);
      write_iter.setAttribute("planeTag", plane.tag);
      write_iter.setAttribute<std::int64_t>("planeNormalAxis",
                                            plane.normal_axis);
      write_iter.setAttribute<double>("planeElevation",
                                      double(plane.elevation));

      std::vector<std::string> patch_suffixes(npatches);
      for (int p = 0; p < npatches; ++p) {
        std::ostringstream buf;
        buf << "_patch" << setw(2) << setfill('0') << p;
        patch_suffixes[p] = buf.str();
      }
      write_iter.setAttribute("patchSuffixes", patch_suffixes);

      for (const auto &patchdata : ghext->patchdata) {
        const int p = patchdata.patch;
        const int nlevels = patchdata.leveldata.size();
        write_iter.setAttribute<std::int64_t>("numLevels" + patch_suffixes[p],
                                              nlevels);

        std::vector<std::string> level_suffixes(nlevels);
        for (int l = 0; l < nlevels; ++l) {
          std::ostringstream buf;
          buf << patch_suffixes[p] << "_lev" << setw(2) << setfill('0') << l;
          level_suffixes[l] = buf.str();
        }
        write_iter.setAttribute("levelSuffixes" + patch_suffixes[p],
                                level_suffixes);

        for (const auto &leveldata : patchdata.leveldata) {
          const int l = leveldata.level;
          const amrex::Geometry &geom = patchdata.amrcore->Geom(l);
          const amrex::Real *const x0 = geom.ProbLo();
          const amrex::Real *const dx = geom.CellSize();

          std::vector<std::int64_t> chunk_infos;
          if (patchdata.is_cartesian) {
            const amrex::FabArrayBase &mfab = *leveldata.fab;
            const int nchunks = mfab.size();
            for (int c = 0; c < nchunks; ++c) {
              const amrex::Box &box = mfab.box(c);
              const int sb = box.smallEnd(plane.normal_axis);
              const int eb = box.bigEnd(plane.normal_axis);
              const double w_lo = double(x0[plane.normal_axis]) +
                                  double(sb) * double(dx[plane.normal_axis]);
              const double w_hi =
                  double(x0[plane.normal_axis]) +
                  double(eb + 1) * double(dx[plane.normal_axis]);
              if (double(plane.elevation) < w_lo ||
                  double(plane.elevation) >= w_hi)
                continue;
              chunk_infos.push_back(box.smallEnd(axis_b));
              chunk_infos.push_back(box.smallEnd(axis_a));
              chunk_infos.push_back(box.bigEnd(axis_b) + 1);
              chunk_infos.push_back(box.bigEnd(axis_a) + 1);
            }
          }
          // Only write chunkInfo when this (patch, level) actually intersects
          // the plane: an empty (zero-length) array attribute cannot be read
          // back by some openPMD/ADIOS2 versions. Readers treat an absent
          // chunkInfo as "no chunks on this level".
          if (!chunk_infos.empty())
            write_iter.setAttribute("chunkInfo" + level_suffixes[l],
                                    chunk_infos);
          write_iter.setAttribute("iteration_num" + level_suffixes[l],
                                  std::int64_t(leveldata.iteration.num));
          write_iter.setAttribute("iteration_den" + level_suffixes[l],
                                  std::int64_t(leveldata.iteration.den));
        }
      }
    }

    bool any_slab_emitted = false;

    for (const auto &patchdata : ghext->patchdata) {
      if (!patchdata.is_cartesian) {
        if (warned_noncart_patches.insert(patchdata.patch).second)
          CCTK_VWARN(CCTK_WARN_ALERT,
                     "OutputOpenPMDPlanes: skipping non-Cartesian patch %d",
                     patchdata.patch);
        continue;
      }

      for (const auto &leveldata : patchdata.leveldata) {
        const amrex::Geometry &geom = patchdata.amrcore->Geom(leveldata.level);
        const amrex::Real *const xlo = geom.ProbLo();
        const amrex::Real *const xhi = geom.ProbHi();
        const amrex::Real *const dx = geom.CellSize();
        const amrex::Box &dom = geom.Domain();
        const amrex::IntVect &ilo3 = dom.smallEnd();
        const amrex::IntVect &ihi3 = dom.bigEnd();

        const std::array<int, 2> idom_lo = {ilo3[axis_a], ilo3[axis_b]};
        const std::array<int, 2> idom_hi = {ihi3[axis_a] + 1 + 1,
                                            ihi3[axis_b] + 1 + 1};
        const std::array<int, 2> idom_shape = {idom_hi[0] - idom_lo[0],
                                               idom_hi[1] - idom_lo[1]};

        const std::array<double, 2> rdom_lo = {double(xlo[axis_a]),
                                               double(xlo[axis_b])};
        const std::array<double, 2> rdom_hi = {double(xhi[axis_a]),
                                               double(xhi[axis_b])};

        const openPMD::Extent extent = {std::uint64_t(idom_shape[1]),
                                        std::uint64_t(idom_shape[0])};

        for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
          if (!output_group.at(gi))
            continue;
          if (CCTK_GroupTypeI(gi) != CCTK_GF)
            continue;

          const auto &groupdata = *leveldata.groupdata.at(gi);
          if (groupdata.mfab.empty())
            continue;
          const int numvars = groupdata.numvars;
          const int tl = 0;
          const amrex::MultiFab &mfab = *groupdata.mfab[tl];
          const amrex::IndexType &indextype = mfab.ixType();

          const int slice_idx = snap_to_grid_index(plane, geom, indextype);
          if (slice_idx < 0)
            continue;

          // snap_to_grid_index only checks the level's full (refined) domain,
          // so a finer level whose boxes don't actually reach the plane still
          // passes. Skip such a (patch, level, group) entirely: defining an
          // openPMD mesh with no stored chunk makes openpmd-api emit "No extent
          // found" read warnings (the mesh has no data). The BoxArray is
          // replicated on every rank, so all ranks agree -- safe since mesh
          // creation is collective. This mirrors the empty-chunkInfo guard.
          bool level_intersects = false;
          for (int c = 0, nc = mfab.size(); c < nc; ++c) {
            const amrex::Box &b = mfab.box(c); // interior box
            if (slice_idx >= b.smallEnd(plane.normal_axis) &&
                slice_idx <= b.bigEnd(plane.normal_axis)) {
              level_intersects = true;
              break;
            }
          }
          if (!level_intersects)
            continue;

          const bool cv_a = indextype.cellCentered(axis_a);
          const bool cv_b = indextype.cellCentered(axis_b);

          const std::string base_meshname =
              make_meshname(gi, patchdata.patch, leveldata.level, tl);
          const std::string meshname = plane.tag + "_" + base_meshname;
          assert(!write_iter.meshes.contains(meshname));
          openPMD::Mesh mesh = write_iter.meshes[meshname];

          mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
          const std::array<const char *, 3> axis_names = {"x", "y", "z"};
          mesh.setAxisLabels(std::vector<std::string>{axis_names.at(axis_b),
                                                      axis_names.at(axis_a)});
          mesh.setGridSpacing(std::vector<double>{
              (rdom_hi[1] - rdom_lo[1]) / double(idom_shape[1] - 1),
              (rdom_hi[0] - rdom_lo[0]) / double(idom_shape[0] - 1)});
          mesh.setGridGlobalOffset(std::vector<double>{rdom_lo[1], rdom_lo[0]});
          mesh.setGridUnitSI(Unit::length);
          mesh.setTimeOffset(CCTK_REAL(0));

          // Record the true location of this mesh's data along the normal
          // axis (the snapped, centering-dependent world coordinate), so
          // readers need not replay the snap rule. The iteration-level
          // planeElevation remains the *requested* (rounded) elevation.
          mesh.setAttribute("planeNormalAxis", std::int64_t(plane.normal_axis));
          mesh.setAttribute("planeIndex", std::int64_t(slice_idx));
          mesh.setAttribute("planeCoordinate", double(snapped_plane_coordinate(
                                                   plane, geom, indextype)));

          const std::vector<double> position = {cv_b ? 0.5 : 0.0,
                                                cv_a ? 0.5 : 0.0};

          const openPMD::Datatype datatype =
              openPMD::determineDatatype<CCTK_REAL>();
          const openPMD::Dataset dataset(datatype, extent);

          std::vector<openPMD::MeshRecordComponent> record_components;
          record_components.reserve(numvars);
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string componentname = make_componentname(gi, vi);
            record_components.push_back(mesh[componentname]);
            auto &rc = record_components.back();
            rc.setPosition(position);
            rc.resetDataset(dataset);
          }

          const int num_local_components = mfab.local_size();
          for (int local_component = 0; local_component < num_local_components;
               ++local_component) {
            const int component = mfab.IndexArray().at(local_component);
            const amrex::Box &validbox = mfab.box(component); // interior

            if (slice_idx < validbox.smallEnd(plane.normal_axis) ||
                slice_idx > validbox.bigEnd(plane.normal_axis))
              continue;

            any_slab_emitted = true;

            // Stage the interior slab in a single pass, straight from the
            // FAB (no intermediate full-ghost copy).
            const amrex::FArrayBox &fab = mfab[component];
            const int na_valid = validbox.length(axis_a);
            const int nb_valid = validbox.length(axis_b);
            const std::ptrdiff_t np_valid = std::ptrdiff_t(na_valid) * nb_valid;

            std::shared_ptr<CCTK_REAL> chunk_ptr(
                new CCTK_REAL[std::size_t(numvars) * np_valid],
                std::default_delete<CCTK_REAL[]>());
            extract_slab_into(fab, plane.normal_axis, slice_idx, numvars,
                              validbox, chunk_ptr.get());

            assert(validbox.smallEnd(axis_a) >= idom_lo[0]);
            assert(validbox.smallEnd(axis_b) >= idom_lo[1]);
            const openPMD::Offset start = {
                std::uint64_t(validbox.smallEnd(axis_b) - idom_lo[1]),
                std::uint64_t(validbox.smallEnd(axis_a) - idom_lo[0])};
            const openPMD::Extent count = {std::uint64_t(nb_valid),
                                           std::uint64_t(na_valid)};
            assert(start[0] + count[0] <= extent[0]);
            assert(start[1] + count[1] <= extent[1]);

            for (int vi = 0; vi < numvars; ++vi) {
              std::shared_ptr<CCTK_REAL> vi_ptr(chunk_ptr, chunk_ptr.get() +
                                                               vi * np_valid);
              record_components.at(vi).storeChunk(vi_ptr, start, count);
            }
          }
        }
      }
    }

    // The geometric pre-check above already wrote no file for a plane outside
    // every level, so reaching here means the plane is in domain. Since
    // any_slab_emitted is per-rank, a rank that owns no intersecting box
    // legitimately emits nothing; only warn if NO rank emitted anything (a real
    // inconsistency), and only from the I/O rank to avoid duplicates.
    {
      int local_emitted = any_slab_emitted ? 1 : 0;
      int global_emitted = 0;
      MPI_Allreduce(&local_emitted, &global_emitted, 1, MPI_INT, MPI_LOR,
                    MPI_COMM_WORLD);
      if (!global_emitted && myproc == ioproc &&
          warned_outside_domain.insert(plane.tag).second)
        CCTK_VWARN(CCTK_WARN_ALERT,
                   "OutputOpenPMDPlanes: plane %s produced no data on any rank "
                   "this iteration",
                   plane.tag.c_str());
    }

    plane_series.flush();
    write_iter.close();

    if (myproc == ioproc) {
      std::ostringstream buf;
      buf << output_dir << "/" << output_file << "." << plane.tag
          << ".openpmd.visit";
      const std::string visitname = buf.str();
      std::ofstream visit(visitname, std::ios::app);
      assert(visit.good());
      visit << output_file << "." << plane.tag << ".it" << setw(8)
            << setfill('0') << cctk_iteration << openPMD::suffix(format)
            << "\n";
    }
  }

  if (io_verbose)
    timer.print();
}

} // namespace PlanesX

#endif // #ifdef HAVE_CAPABILITY_openPMD_api
