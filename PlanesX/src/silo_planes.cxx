#include "silo_planes.hxx"

#include "carpetx_params.hxx"
#include "planes.hxx"

#include "driver.hxx"
#include "timer.hxx"
// Not exported by CarpetX's interface.ccl; include via the arrangement path
// (same idiom as the IOUtil header below).
#include <CarpetX/CarpetX/src/io_meta.hxx>
#include <CarpetX/CarpetX/src/mpi_types.hxx>

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Parameters.h>

#ifdef HAVE_CAPABILITY_Silo

#include <AMReX.H>
#include <AMReX_IntVect.H>

#include <mpi.h>

#include <silo.hxx>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace PlanesX {
using namespace CarpetX;

namespace {

constexpr int planes_ndims = 2;

template <typename T> struct db_datatype;
template <> struct db_datatype<char> : std::integral_constant<int, DB_CHAR> {};
template <> struct db_datatype<int> : std::integral_constant<int, DB_INT> {};
template <> struct db_datatype<long> : std::integral_constant<int, DB_LONG> {};
template <>
struct db_datatype<long long> : std::integral_constant<int, DB_LONG_LONG> {};
template <>
struct db_datatype<float> : std::integral_constant<int, DB_FLOAT> {};
template <>
struct db_datatype<double> : std::integral_constant<int, DB_DOUBLE> {};
template <typename T> constexpr int db_datatype_v = db_datatype<T>::value;

// Centering-independent ownership for interior-only slabs: the cell whose
// centre is nearest the plane, clamped into the domain (a domain-face vertex
// plane keeps an owner). Exactly one interior box contains this cell, for
// every centering.
inline int plane_owner_cell(const plane_spec_t &plane,
                            const amrex::Geometry &geom) {
  const int n = plane.normal_axis;
  const double r =
      (double(plane.elevation) - geom.ProbLo()[n]) / geom.CellSize()[n];
  const amrex::Box &dom = geom.Domain();
  const int oc = dom.smallEnd(n) + int(std::lround(r - 0.5));
  return std::max(dom.smallEnd(n), std::min(dom.bigEnd(n), oc));
}

struct plane_mesh_props_t {
  std::array<int, 2> nghosts;
  std::string centering_tag;
  friend bool operator<(const plane_mesh_props_t &p,
                        const plane_mesh_props_t &q) {
    if (p.nghosts != q.nghosts)
      return p.nghosts < q.nghosts;
    return p.centering_tag < q.centering_tag;
  }
};

std::string make_plane_subdirname(const std::string &file_name,
                                  const std::string &plane_tag, int iteration) {
  std::ostringstream buf;
  buf << file_name << "." << plane_tag << ".it" << std::setw(8)
      << std::setfill('0') << iteration << ".silo_planes.dir";
  return buf.str();
}

std::string make_plane_filename(const std::string &file_name,
                                const std::string &plane_tag, int iteration,
                                int ioserver = -1) {
  std::ostringstream buf;
  buf << file_name << "." << plane_tag << ".it" << std::setw(8)
      << std::setfill('0') << iteration;
  if (ioserver >= 0)
    buf << ".p" << std::setw(6) << std::setfill('0') << ioserver;
  buf << ".silo";
  return buf.str();
}

std::string make_plane_meshname(const std::string &plane_tag,
                                const std::string &centering_tag,
                                const std::array<int, 2> &nghosts,
                                int patch = -1, int reflevel = -1,
                                int component = -1) {
  assert((patch == -1) == (reflevel == -1));
  assert((patch == -1) == (component == -1));
  std::ostringstream buf;
  buf << (patch >= 0 ? "box" : "gh") << "." << plane_tag;
  if (!centering_tag.empty())
    buf << "." << centering_tag;
  buf << ".ghosts";
  for (int d = 0; d < 2; ++d) {
    if (d > 0)
      buf << "_";
    buf << std::setw(2) << std::setfill('0') << nghosts[d];
  }
  if (patch >= 0)
    buf << ".m" << std::setw(4) << std::setfill('0') << patch << ".rl"
        << std::setw(2) << std::setfill('0') << reflevel << ".c" << std::setw(8)
        << std::setfill('0') << component;
  return DB::legalize_name(buf.str());
}

std::string make_plane_varname(int gi, int vi, const std::string &plane_tag,
                               int patch = -1, int reflevel = -1,
                               int component = -1) {
  assert((patch == -1) == (reflevel == -1));
  assert((patch == -1) == (component == -1));
  std::string varname;
  {
    assert(vi >= 0);
    const int v0 = CCTK_FirstVarIndexI(gi);
    varname = CCTK_FullVarName(v0 + vi);
    static const std::regex colons_re("::");
    varname = std::regex_replace(varname, colons_re, "-");
    for (auto &ch : varname)
      ch = std::tolower(static_cast<unsigned char>(ch));
  }
  std::ostringstream buf;
  buf << varname << "." << plane_tag;
  if (reflevel >= 0)
    buf << ".m" << std::setw(4) << std::setfill('0') << patch << ".rl"
        << std::setw(2) << std::setfill('0') << reflevel << ".c" << std::setw(8)
        << std::setfill('0') << component;
  return DB::legalize_name(buf.str());
}

// --- Nameschemes (silo_planes_nameschemes) ----------------------------------
//
// A Silo namescheme replaces a multi-block object's per-block string list
// with one printf-like pattern ("|<pattern>|<expr>...") evaluated per block
// index n; "#/<array>[n]" references an int array DBWrite-n into the same
// file. The block -> (patch, level, component, file) mapping is irregular
// (only intersecting components are listed), so external index arrays are
// required. The expanded names must equal the explicit leaf names
// byte-for-byte; rather than deriving the pattern correct-by-construction,
// the writer expands every written namescheme through the same file-based
// DBMakeNamescheme path VisIt uses and compares against the explicit list --
// on any mismatch (or constructor failure) it falls back to writing the
// explicit list, so a namescheme only ever lands in the file after being
// proven equivalent.

// The trailing index fields of the (legalized) per-block leaf object names
// for block (patch, level, component) = (0, 0, 0).
const char *const block000_suffix = "_m0000_rl00_c00000000";

// True iff the pair (file namescheme, block namescheme) -- whose "#/..."
// arrays must already be DBWritten to `file` -- expands, for every index, to
// the explicit "<file>:<object>" name in `expected`.
bool nameschemes_match(DBfile *const file, const std::string &file_ns,
                       const std::string &block_ns,
                       const std::vector<std::string> &expected) {
  DBnamescheme *const fns = DBMakeNamescheme(file_ns.c_str(), 0, file, 0);
  DBnamescheme *const bns = DBMakeNamescheme(block_ns.c_str(), 0, file, 0);
  bool ok = fns && bns;
  for (std::size_t i = 0; i < expected.size() && ok; ++i) {
    // Copy: DBGetName returns a pointer into a small ring of static buffers.
    const char *const f = DBGetName(fns, int(i));
    const std::string fname = f ? f : "";
    const char *const b = DBGetName(bns, int(i));
    ok = b && expected[i] == fname + ":" + b;
  }
  if (fns)
    DBFreeNamescheme(fns);
  if (bns)
    DBFreeNamescheme(bns);
  return ok;
}

// Record the true location of the plane data in `file` (a per-plane Silo
// file), so readers need not replay the snap rule: "plane_normal_axis" (int),
// "plane_elevation" (the requested, rounded elevation), and per Cartesian
// patch the snapped world coordinate per level and normal-axis centering,
// "plane_ncoord_vertex_m%04d" / "plane_ncoord_cell_m%04d" (double[nlevels],
// NaN where the plane misses the level or the patch is non-Cartesian).
void write_plane_location(DBfile *const file, const plane_spec_t &plane) {
  using namespace CarpetX;
  int ierr;
  {
    const int dims = 1;
    const int axis = plane.normal_axis;
    ierr = DBWrite(file, "plane_normal_axis", &axis, &dims, 1, DB_INT);
    assert(!ierr);
    const double elevation = double(plane.elevation);
    ierr = DBWrite(file, "plane_elevation", &elevation, &dims, 1, DB_DOUBLE);
    assert(!ierr);
  }
  for (const auto &patchdata : ghext->patchdata) {
    const int nlevels = int(patchdata.leveldata.size());
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> coord_vertex(nlevels, nan), coord_cell(nlevels, nan);
    if (patchdata.is_cartesian) {
      for (const auto &leveldata : patchdata.leveldata) {
        const amrex::Geometry &geom = patchdata.amrcore->Geom(leveldata.level);
        coord_vertex.at(leveldata.level) = double(snapped_plane_coordinate(
            plane, geom, amrex::IndexType::TheNodeType()));
        coord_cell.at(leveldata.level) = double(snapped_plane_coordinate(
            plane, geom, amrex::IndexType::TheCellType()));
      }
    }
    std::ostringstream vbuf, cbuf;
    vbuf << "plane_ncoord_vertex_m" << std::setw(4) << std::setfill('0')
         << patchdata.patch;
    cbuf << "plane_ncoord_cell_m" << std::setw(4) << std::setfill('0')
         << patchdata.patch;
    const int dims = nlevels;
    ierr = DBWrite(file, vbuf.str().c_str(), coord_vertex.data(), &dims, 1,
                   DB_DOUBLE);
    assert(!ierr);
    ierr = DBWrite(file, cbuf.str().c_str(), coord_cell.data(), &dims, 1,
                   DB_DOUBLE);
    assert(!ierr);
  }
}

} // namespace

void OutputSiloPlanes(const cGH *const cctkGH,
                      const std::vector<bool> &output_group,
                      const std::vector<plane_spec_t> &planes,
                      const std::string &output_dir,
                      const std::string &output_file) {
  // out_mode and out_proc_every are IO:: parameters shared into PlanesX's
  // param.ccl (CarpetX shares them the same way).
  DECLARE_CCTK_PARAMETERS;

  const int cctk_iteration = cctkGH->cctk_iteration;
  const CCTK_REAL cctk_time = cctkGH->cctk_time;

  // CarpetX's (private) Silo compression parameter; PlanesX follows it.
  const char *const out_silo_compression_options =
      get_carpetx_string_param("out_silo_compression_options");

  int ierr;

  static Timer timer("OutputSiloPlanes");
  Interval interval(timer);

  if (planes.empty())
    return;
  if (std::count(output_group.begin(), output_group.end(), true) == 0)
    return;

  const MPI_Comm mpi_comm = MPI_COMM_WORLD;
  const int myproc = CCTK_MyProc(cctkGH);
  const int nprocs = CCTK_nProcs(cctkGH);

  const int ioproc_every = [&]() {
    if (CCTK_EQUALS(out_mode, "proc"))
      return 1;
    if (CCTK_EQUALS(out_mode, "np"))
      return int(out_proc_every);
    if (CCTK_EQUALS(out_mode, "onefile"))
      return nprocs;
    assert(0);
  }();
  assert(ioproc_every > 0);

  const bool write_file = myproc % ioproc_every == 0;
  const int metafile_ioproc = nprocs == 1 || ioproc_every == 1 ? 0 : 1;
  const bool write_metafile = myproc == metafile_ioproc;

  DBShowErrors(DB_ALL_AND_DRVR, nullptr);
  DBSetCompression(out_silo_compression_options);
  DBSetEnableChecksums(1);

  static std::set<int> warned_noncart_patches;
  static std::set<std::string> warned_outside_domain;

  static std::once_flag create_out_dir;
  std::call_once(create_out_dir, [&]() {
    const int rc = CCTK_CreateDirectory(0755, output_dir.c_str());
    assert(rc >= 0);
  });

  // Parameter provenance goes into the (per-iteration) metafile only: the
  // leaf files of the same iteration would just duplicate it. Serialized once
  // per call.
  std::string all_parameters;
  if (write_metafile) {
    char *const data = IOUtil_GetAllParameters(cctkGH, 1 /*all*/);
    all_parameters = data;
    std::free(data);
  }

  const bool single_precision =
      CCTK_EQUALS(silo_planes_single_precision, "yes") ||
      (CCTK_EQUALS(silo_planes_single_precision, "default") &&
       out_single_precision);

  for (const auto &plane : planes) {
    const auto in_axes = in_plane_axes(plane.normal_axis);
    const int axis_a = in_axes[0], axis_b = in_axes[1];

    // Skip the plane entirely -- creating no directory, data file or metafile
    // -- if its elevation lies outside every Cartesian (patch, level) along the
    // normal axis. The decision is purely geometric, so all ranks agree.
    if (!plane_in_any_domain(plane, output_group)) {
      if (warned_outside_domain.insert(plane.tag).second)
        CCTK_VWARN(CCTK_WARN_ALERT,
                   "OutputSiloPlanes: plane %s lies outside all Cartesian "
                   "(patch, level) extents; writing no file",
                   plane.tag.c_str());
      continue;
    }

    const std::string subdirname =
        make_plane_subdirname(output_file, plane.tag, cctk_iteration);
    const std::string pathname = output_dir + "/" + subdirname;
    ierr = CCTK_CreateDirectory(0755, pathname.c_str());
    assert(ierr >= 0);

    // The leaf file is created lazily, on the first slab this rank actually
    // writes for this plane: an IO rank that owns no intersecting box would
    // otherwise leave a file holding only the plane-location arrays and no
    // mesh/var object, which VisIt's Silo reader rejects at open time ("It
    // may be an invalid file"). With silo_planes_ghosts=no the single-owner
    // membership makes such ranks routine (the production failure mode); with
    // ghosts they occur on any run large enough that some rank's boxes all
    // miss the plane. The metafile references only files of ranks that own a
    // slab, so the never-created files are never referenced.
    DB::ptr<DBfile> file;
    const auto ensure_file = [&]() {
      assert(write_file);
      if (!file) {
        const std::string filename =
            pathname + "/" +
            make_plane_filename(output_file, plane.tag, cctk_iteration,
                                myproc / ioproc_every);
        file = DB::make(DBCreate(filename.c_str(), DB_CLOBBER, DB_LOCAL,
                                 output_file.c_str(), DB_HDF5));
        assert(file);

        write_plane_location(file.get(), plane);
      }
      return file.get();
    };

    for (const auto &patchdata : ghext->patchdata) {
      if (!patchdata.is_cartesian) {
        if (warned_noncart_patches.insert(patchdata.patch).second)
          CCTK_VWARN(CCTK_WARN_ALERT,
                     "OutputSiloPlanes: skipping non-Cartesian patch %d",
                     patchdata.patch);
        continue;
      }

      for (const auto &leveldata : patchdata.leveldata) {
        const amrex::Geometry &geom = patchdata.amrcore->Geom(leveldata.level);
        const amrex::Real *const x0 = geom.ProbLo();
        const amrex::Real *const dx = geom.CellSize();

        std::set<plane_mesh_props_t> have_meshes;
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
          const amrex::DistributionMapping &dm = mfab.DistributionMap();

          const int slice_idx = snap_to_grid_index(plane, geom, indextype);
          if (slice_idx < 0)
            continue;

          // Interior-only slabs use a single, centering-independent OWNER box
          // per plane: a vertex-in-normal plane on a shared box face lies in
          // BOTH adjacent interior boxes while cell-in-normal data lives in
          // one, so per-centering membership would give the shared multimesh
          // a different block count than the cell-centred multivars -- which
          // VisIt rejects. The owner is the cell whose centre is nearest the
          // plane, clamped into the domain so a domain-face vertex plane
          // keeps an owner. (With ghosts, every adjacent exterior box covers
          // the plane for every centering, so membership stays uniform there
          // without this.)
          const int owner_cell = plane_owner_cell(plane, geom);

          const bool cv_a = indextype.cellCentered(axis_a);
          const bool cv_b = indextype.cellCentered(axis_b);
          const int in_plane_rank = int(cv_a) + int(cv_b);
          const bool per_group_mesh = (in_plane_rank == 1);
          const int centering =
              (in_plane_rank == 2) ? DB_ZONECENT : DB_NODECENT;
          const std::string centering_tag =
              per_group_mesh ? (cv_a ? "cv" : "vc") : std::string("");

          const std::array<int, dim> &nghosts3 = groupdata.nghostzones;
          const std::array<int, 2> nghosts_inplane =
              silo_planes_ghosts
                  ? std::array<int, 2>{nghosts3[axis_a], nghosts3[axis_b]}
                  : std::array<int, 2>{0, 0};
          const plane_mesh_props_t mesh_props{nghosts_inplane, centering_tag};
          const bool have_mesh = have_meshes.count(mesh_props);

          const int ncomponents = dm.size();
          for (int component = 0; component < ncomponents; ++component) {
            const int proc = dm[component];
            const int ioproc = proc / ioproc_every * ioproc_every;
            const bool send_this_fab = proc == myproc;
            const bool write_this_fab = ioproc == myproc;
            if (!(send_this_fab || write_this_fab))
              continue;

            // The slab extent: the full exterior box (with ghosts) or, with
            // silo_planes_ghosts=no, the interior box only. Both are
            // replicated on every rank, so sender and receiver agree.
            const amrex::Box databox = silo_planes_ghosts
                                           ? mfab.fabbox(component)
                                           : mfab.box(component);
            if (silo_planes_ghosts) {
              if (slice_idx < databox.smallEnd(plane.normal_axis) ||
                  slice_idx > databox.bigEnd(plane.normal_axis))
                continue;
            } else {
              const amrex::Box cellbox =
                  amrex::enclosedCells(mfab.box(component));
              if (owner_cell < cellbox.smallEnd(plane.normal_axis) ||
                  owner_cell > cellbox.bigEnd(plane.normal_axis))
                continue;
              // The owner box's interior always contains the snapped slab.
              assert(slice_idx >= databox.smallEnd(plane.normal_axis) &&
                     slice_idx <= databox.bigEnd(plane.normal_axis));
            }

            const std::array<int, 2> dims = {databox.length(axis_a),
                                             databox.length(axis_b)};
            const std::ptrdiff_t zonecount = std::ptrdiff_t(dims[0]) * dims[1];
            assert(numvars * zonecount <= INT_MAX);

            std::vector<CCTK_REAL> local_buf;
            std::vector<CCTK_REAL> recv_buf;
            const CCTK_REAL *data = nullptr;

            if (send_this_fab) {
              const amrex::FArrayBox &fab = mfab[component];
              local_buf.resize(std::size_t(numvars) * zonecount);
              extract_slab_into(fab, plane.normal_axis, slice_idx, numvars,
                                databox, local_buf.data());
              if (write_this_fab) {
                data = local_buf.data();
              } else {
                const int mpi_tag = 22910;
                MPI_Send(local_buf.data(), int(numvars * zonecount),
                         mpi_datatype_v<CCTK_REAL>, ioproc, mpi_tag, mpi_comm);
              }
            } else {
              assert(write_this_fab);
              const int mpi_tag = 22910;
              recv_buf.resize(std::size_t(numvars) * zonecount);
              MPI_Recv(recv_buf.data(), int(numvars * zonecount),
                       mpi_datatype_v<CCTK_REAL>, proc, mpi_tag, mpi_comm,
                       MPI_STATUS_IGNORE);
              data = recv_buf.data();
            }

            if (!write_this_fab)
              continue;

            DBfile *const leaf_file = ensure_file();

            const std::string meshname = make_plane_meshname(
                plane.tag, centering_tag, mesh_props.nghosts, patchdata.patch,
                leveldata.level, component);

            // Optional single-precision conversion of the gathered slab.
            std::vector<float> float_buf;
            if (single_precision) {
              const std::ptrdiff_t n = std::ptrdiff_t(numvars) * zonecount;
              float_buf.resize(n);
              for (std::ptrdiff_t k = 0; k < n; ++k)
                float_buf[k] = float(data[k]);
            }

            if (!have_mesh) {
              const std::array<int, 2> dims_vc =
                  per_group_mesh ? dims
                                 : std::array<int, 2>{dims[0] + int(cv_a),
                                                      dims[1] + int(cv_b)};

              std::array<std::vector<CCTK_REAL>, 2> coords;
              std::array<const void *, 2> coord_ptrs;
              for (int d = 0; d < 2; ++d) {
                const int ax = (d == 0) ? axis_a : axis_b;
                const bool ax_cc = (d == 0) ? cv_a : cv_b;
                const double offset = (per_group_mesh && ax_cc) ? 0.5 : 0.0;
                coords[d].resize(dims_vc[d]);
                for (int i = 0; i < dims_vc[d]; ++i)
                  coords[d][i] =
                      x0[ax] + (databox.smallEnd(ax) + i + offset) * dx[ax];
                coord_ptrs[d] = coords[d].data();
              }

              const DB::ptr<DBoptlist> optlist = DB::make(DBMakeOptlist(10));
              assert(optlist);
              int cartesian = DB_CARTESIAN;
              ierr = DBAddOption(optlist.get(), DBOPT_COORDSYS, &cartesian);
              assert(!ierr);
              int cycle = cctk_iteration;
              ierr = DBAddOption(optlist.get(), DBOPT_CYCLE, &cycle);
              assert(!ierr);
              std::array<int, 2> min_index = nghosts_inplane;
              std::array<int, 2> max_index = min_index;
              ierr =
                  DBAddOption(optlist.get(), DBOPT_LO_OFFSET, min_index.data());
              assert(!ierr);
              ierr =
                  DBAddOption(optlist.get(), DBOPT_HI_OFFSET, max_index.data());
              assert(!ierr);
              int column_major = 0;
              ierr =
                  DBAddOption(optlist.get(), DBOPT_MAJORORDER, &column_major);
              assert(!ierr);
              double dtime = cctk_time;
              ierr = DBAddOption(optlist.get(), DBOPT_DTIME, &dtime);
              assert(!ierr);
              int hide_from_gui = 1;
              ierr = DBAddOption(optlist.get(), DBOPT_HIDE_FROM_GUI,
                                 &hide_from_gui);
              assert(!ierr);

              ierr = DBPutQuadmesh(leaf_file, meshname.c_str(), nullptr,
                                   coord_ptrs.data(), dims_vc.data(),
                                   planes_ndims, db_datatype_v<CCTK_REAL>,
                                   DB_COLLINEAR, optlist.get());
              assert(!ierr);
              have_meshes.insert(mesh_props);
            }

            const DB::ptr<DBoptlist> var_optlist = DB::make(DBMakeOptlist(10));
            assert(var_optlist);
            int cartesian = DB_CARTESIAN;
            ierr = DBAddOption(var_optlist.get(), DBOPT_COORDSYS, &cartesian);
            assert(!ierr);
            int cycle = cctk_iteration;
            ierr = DBAddOption(var_optlist.get(), DBOPT_CYCLE, &cycle);
            assert(!ierr);
            int column_major = 0;
            ierr =
                DBAddOption(var_optlist.get(), DBOPT_MAJORORDER, &column_major);
            assert(!ierr);
            double dtime = cctk_time;
            ierr = DBAddOption(var_optlist.get(), DBOPT_DTIME, &dtime);
            assert(!ierr);
            int hide_from_gui = 1;
            ierr = DBAddOption(var_optlist.get(), DBOPT_HIDE_FROM_GUI,
                               &hide_from_gui);
            assert(!ierr);

            for (int vi = 0; vi < numvars; ++vi) {
              const std::string varname =
                  make_plane_varname(gi, vi, plane.tag, patchdata.patch,
                                     leveldata.level, component);
              const void *const data_ptr =
                  single_precision
                      ? static_cast<const void *>(float_buf.data() +
                                                  vi * zonecount)
                      : static_cast<const void *>(data + vi * zonecount);
              ierr = DBPutQuadvar1(
                  leaf_file, varname.c_str(), meshname.c_str(), data_ptr,
                  dims.data(), planes_ndims, nullptr, 0,
                  single_precision ? DB_FLOAT : db_datatype_v<CCTK_REAL>,
                  centering, var_optlist.get());
              assert(!ierr);
            }
          }
        }
      }
    }

    // The geometric pre-check above already wrote no file for a plane outside
    // every level, so reaching here means the plane is in domain. The leaf
    // file exists iff this rank emitted a slab (lazy creation above); a rank
    // that owns no intersecting box legitimately emits nothing. Only warn if
    // NO rank emitted anything (a real inconsistency), and only from the
    // metafile rank to avoid duplicates.
    {
      int local_emitted = file ? 1 : 0;
      int global_emitted = 0;
      MPI_Allreduce(&local_emitted, &global_emitted, 1, MPI_INT, MPI_LOR,
                    mpi_comm);
      if (!global_emitted && write_metafile &&
          warned_outside_domain.insert(plane.tag).second)
        CCTK_VWARN(CCTK_WARN_ALERT,
                   "OutputSiloPlanes: plane %s produced no data on any rank "
                   "this iteration",
                   plane.tag.c_str());
    }

    if (write_metafile) {
      const std::string metafilename =
          output_dir + "/" +
          make_plane_filename(output_file, plane.tag, cctk_iteration);
      const DB::ptr<DBfile> metafile =
          DB::make(DBCreate(metafilename.c_str(), DB_CLOBBER, DB_LOCAL,
                            output_file.c_str(), DB_HDF5));
      assert(metafile);

      {
        const int pdims = all_parameters.length();
        ierr = DBWrite(metafile.get(), "AllParameters", all_parameters.data(),
                       &pdims, 1, DB_CHAR);
        assert(!ierr);
      }

      {
        const int mdims = 1;
        const int value = 1;
        ierr = DBWrite(metafile.get(), "MetadataIsTimeVarying", &value, &mdims,
                       1, DB_INT);
        assert(!ierr);
      }

      write_plane_location(metafile.get(), plane);

      // Namescheme pattern builders: the explicit name for block index
      // (patch, level, component) = (0, 0, 0) with its trailing index fields
      // replaced by printf conversions, plus "#/<prefix>_ns_*[n]" external
      // index arrays. Any construction subtlety (legalization, '%' or '|'
      // inside a name) is caught by the expansion check before use, which
      // then falls back to explicit name lists.
      const auto make_file_ns = [&](const std::string &arrays_prefix) {
        std::string f0 =
            subdirname + "/" +
            make_plane_filename(output_file, plane.tag, cctk_iteration, 0);
        f0.replace(f0.size() - std::strlen(".p000000.silo"), std::string::npos,
                   ".p%06d.silo");
        return "|" + f0 + "|#/" + arrays_prefix + "_ns_file[n]";
      };
      const auto make_block_ns = [&](std::string name000,
                                     const std::string &arrays_prefix) {
        name000.replace(name000.size() - std::strlen(block000_suffix),
                        std::string::npos, "_m%04d_rl%02d_c%08d");
        return "|" + name000 + "|#/" + arrays_prefix + "_ns_patch[n]|#/" +
               arrays_prefix + "_ns_level[n]|#/" + arrays_prefix +
               "_ns_comp[n]";
      };

      // Per mesh_props: the block count its multimesh was written with when
      // nameschemes were used, 0 when explicit name lists were written.
      std::map<plane_mesh_props_t, int> meta_meshes;
      for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
        if (!output_group.at(gi))
          continue;
        if (CCTK_GroupTypeI(gi) != CCTK_GF)
          continue;

        const auto &patchdata0 = ghext->patchdata.at(0);
        const auto &leveldata0 = patchdata0.leveldata.at(0);
        const auto &groupdata0 = *leveldata0.groupdata.at(gi);
        if (groupdata0.mfab.empty())
          continue;
        const int numvars = groupdata0.numvars;
        const int tl = 0;
        const amrex::MultiFab &mfab0 = *groupdata0.mfab[tl];
        const amrex::IndexType &indextype = mfab0.ixType();

        const bool cv_a = indextype.cellCentered(axis_a);
        const bool cv_b = indextype.cellCentered(axis_b);
        const std::string centering_tag =
            (int(cv_a) + int(cv_b) == 1)
                ? (cv_a ? std::string("cv") : std::string("vc"))
                : std::string("");

        const std::array<int, dim> &nghosts3 = groupdata0.nghostzones;
        const std::array<int, 2> nghosts_inplane =
            silo_planes_ghosts
                ? std::array<int, 2>{nghosts3[axis_a], nghosts3[axis_b]}
                : std::array<int, 2>{0, 0};
        const plane_mesh_props_t mesh_props{nghosts_inplane, centering_tag};

        const int nlevels = ghext->num_levels();
        const int npatches = ghext->num_patches();

        struct slab_t {
          int patch, level, component, proc;
          std::array<int, 2> ilo, ihi;
          std::array<int, 2> interior_ilo, interior_ihi;
          std::array<double, 2> xlo, xhi;
          int zonecount;
        };
        std::vector<slab_t> slabs;
        std::vector<int> comp0_level(nlevels, 0);
        std::vector<int> ncomps_level(nlevels, 0);
        std::vector<std::vector<int> > comp0_level_patch(
            nlevels, std::vector<int>(npatches, 0));
        std::vector<std::vector<int> > ncomps_level_patch(
            nlevels, std::vector<int>(npatches, 0));

        std::vector<std::string> meshnames;
        std::vector<std::vector<std::string> > varnames_per_var(numvars);

        for (int level = 0; level < nlevels; ++level) {
          comp0_level[level] = int(slabs.size());
          for (int patch = 0; patch < npatches; ++patch) {
            const auto &patchdata = ghext->patchdata.at(patch);
            if (!patchdata.is_cartesian)
              continue;
            if (level >= int(patchdata.leveldata.size()))
              continue;
            const auto &leveldata = patchdata.leveldata.at(level);
            const amrex::Geometry &geom = patchdata.amrcore->Geom(level);
            const int slice_idx = snap_to_grid_index(plane, geom, indextype);
            if (slice_idx < 0)
              continue;

            const auto &gdata = *leveldata.groupdata.at(gi);
            if (gdata.mfab.empty())
              continue;
            const amrex::MultiFab &mfab = *gdata.mfab[tl];
            const amrex::DistributionMapping &dm = mfab.DistributionMap();
            const int ncomponents = dm.size();
            comp0_level_patch[level][patch] = int(slabs.size());

            const amrex::Real *const x0 = geom.ProbLo();
            const amrex::Real *const dx = geom.CellSize();

            // Mirror the leaf writer's centering-independent ownership for
            // interior-only slabs (see the leaf pass for why).
            const int owner_cell = plane_owner_cell(plane, geom);

            for (int c = 0; c < ncomponents; ++c) {
              // Mirror the leaf writer's slab extent (exterior or interior).
              const amrex::Box databox =
                  silo_planes_ghosts ? mfab.fabbox(c) : mfab.box(c);
              if (silo_planes_ghosts) {
                if (slice_idx < databox.smallEnd(plane.normal_axis) ||
                    slice_idx > databox.bigEnd(plane.normal_axis))
                  continue;
              } else {
                const amrex::Box cellbox = amrex::enclosedCells(mfab.box(c));
                if (owner_cell < cellbox.smallEnd(plane.normal_axis) ||
                    owner_cell > cellbox.bigEnd(plane.normal_axis))
                  continue;
              }
              const amrex::Box &validbox = mfab.box(c);

              slab_t s;
              s.patch = patch;
              s.level = level;
              s.component = c;
              s.proc = dm[c];
              s.ilo = {databox.smallEnd(axis_a), databox.smallEnd(axis_b)};
              s.ihi = {databox.bigEnd(axis_a), databox.bigEnd(axis_b)};
              s.interior_ilo = {validbox.smallEnd(axis_a),
                                validbox.smallEnd(axis_b)};
              s.interior_ihi = {validbox.bigEnd(axis_a),
                                validbox.bigEnd(axis_b)};
              s.xlo = {
                  double(x0[axis_a] + databox.smallEnd(axis_a) * dx[axis_a]),
                  double(x0[axis_b] + databox.smallEnd(axis_b) * dx[axis_b])};
              s.xhi = {
                  double(x0[axis_a] + databox.bigEnd(axis_a) * dx[axis_a]),
                  double(x0[axis_b] + databox.bigEnd(axis_b) * dx[axis_b])};
              const int len_a = databox.length(axis_a);
              const int len_b = databox.length(axis_b);
              s.zonecount = (len_a + int(cv_a)) * (len_b + int(cv_b));
              slabs.push_back(s);

              const std::string proc_filename =
                  subdirname + "/" +
                  make_plane_filename(output_file, plane.tag, cctk_iteration,
                                      s.proc / ioproc_every);
              meshnames.push_back(proc_filename + ":" +
                                  make_plane_meshname(plane.tag, centering_tag,
                                                      mesh_props.nghosts, patch,
                                                      level, c));
              for (int vi = 0; vi < numvars; ++vi)
                varnames_per_var[vi].push_back(
                    proc_filename + ":" +
                    make_plane_varname(gi, vi, plane.tag, patch, level, c));
            }
            ncomps_level_patch[level][patch] =
                int(slabs.size()) - comp0_level_patch[level][patch];
          }
          ncomps_level[level] = int(slabs.size()) - comp0_level[level];
        }

        if (slabs.empty())
          continue;
        const int ncomps_total = int(slabs.size());

        if (!meta_meshes.count(mesh_props)) {
          const std::string multimeshname =
              make_plane_meshname(plane.tag, centering_tag, mesh_props.nghosts);
          const std::string levelmaps_name =
              multimeshname + "_wmrgtree_lvlMaps";
          const std::string childmaps_name =
              multimeshname + "_wmrgtree_chldMaps";
          // Per-multimesh tree name: one metafile holds several multimeshes
          // (one per centering), so a shared "mrgTree" would be overwritten and
          // the earlier multimeshes would reference another centering's maps.
          const std::string mrgtree_name = multimeshname + "_mrgTree";

          // Map each component to its finer-level children. The AMR mrgtree is
          // emitted only when some parent->child relationship exists: otherwise
          // (single level, or a plane crossing one level) the child map is all
          // empty, Silo stores no segment-data array, and VisIt crashes freeing
          // it (DBFreeGroupelmap on a NULL array). A non-AMR dataset is written
          // as a plain multimesh. See PlanesX/doc/plane_output.md.
          std::vector<std::vector<int> > child_data(ncomps_total);
          for (int idx = 0; idx < ncomps_total; ++idx) {
            const auto &s = slabs[idx];
            const int fine_level = s.level + 1;
            if (fine_level >= nlevels)
              continue;
            const int fine_comp0 = comp0_level_patch[fine_level][s.patch];
            const int fine_ncomps = ncomps_level_patch[fine_level][s.patch];
            const std::array<int, 2> ref_lo = {2 * s.interior_ilo[0],
                                               2 * s.interior_ilo[1]};
            const std::array<int, 2> ref_hi = {2 * s.interior_ihi[0] + 1,
                                               2 * s.interior_ihi[1] + 1};
            for (int fi = 0; fi < fine_ncomps; ++fi) {
              const auto &fs = slabs[fine_comp0 + fi];
              if (fs.interior_ihi[0] >= ref_lo[0] &&
                  fs.interior_ilo[0] <= ref_hi[0] &&
                  fs.interior_ihi[1] >= ref_lo[1] &&
                  fs.interior_ilo[1] <= ref_hi[1])
                child_data[idx].push_back(fine_comp0 + fi);
            }
          }
          std::vector<int> num_children(ncomps_total);
          int total_children = 0;
          for (int idx = 0; idx < ncomps_total; ++idx) {
            num_children[idx] = int(child_data[idx].size());
            total_children += num_children[idx];
          }
          const bool emit_amr = total_children > 0;

          if (emit_amr) {
            {
              std::vector<int> segment_types(nlevels, DB_BLOCKCENT);
              std::vector<std::vector<int> > segment_data(nlevels);
              for (int l = 0; l < nlevels; ++l) {
                segment_data[l].reserve(ncomps_level[l]);
                for (int c = 0; c < ncomps_level[l]; ++c)
                  segment_data[l].push_back(comp0_level[l] + c);
              }
              std::vector<const int *> segment_data_ptrs;
              segment_data_ptrs.reserve(nlevels);
              for (const auto &d : segment_data)
                segment_data_ptrs.push_back(d.data());
              // The per-level segment lengths are by construction the entries
              // of ncomps_level.
              ierr = DBPutGroupelmap(
                  metafile.get(), levelmaps_name.c_str(), nlevels,
                  segment_types.data(), ncomps_level.data(), nullptr,
                  segment_data_ptrs.data(), nullptr, 0, nullptr);
              assert(!ierr);
            }

            {
              std::vector<int> segment_types(ncomps_total, DB_BLOCKCENT);
              std::vector<const int *> segment_data_ptrs;
              segment_data_ptrs.reserve(ncomps_total);
              for (const auto &d : child_data)
                segment_data_ptrs.push_back(d.data());
              ierr = DBPutGroupelmap(
                  metafile.get(), childmaps_name.c_str(), ncomps_total,
                  segment_types.data(), num_children.data(), nullptr,
                  segment_data_ptrs.data(), nullptr, 0, nullptr);
              assert(!ierr);
            }

            {
              const int max_children = 2;
              const DB::ptr<DBmrgtree> mrgtree = DB::make(
                  DBMakeMrgtree(DB_MULTIMESH, 0, max_children, nullptr));
              assert(mrgtree);
              ierr =
                  DBAddRegion(mrgtree.get(), "amr_decomp", 0, max_children,
                              nullptr, 0, nullptr, nullptr, nullptr, nullptr);
              assert(!ierr);
              ierr = DBSetCwr(mrgtree.get(), "amr_decomp");
              assert(ierr >= 0);

              {
                ierr = DBAddRegion(mrgtree.get(), "levels", 0, nlevels, nullptr,
                                   0, nullptr, nullptr, nullptr, nullptr);
                assert(!ierr);
                ierr = DBSetCwr(mrgtree.get(), "levels");
                assert(ierr >= 0);
                const std::vector<std::string> region_names{"@level%d@n"};
                std::vector<const char *> region_name_ptrs;
                region_name_ptrs.reserve(region_names.size());
                for (const auto &n : region_names)
                  region_name_ptrs.push_back(n.c_str());
                std::vector<int> segment_ids(nlevels);
                std::vector<int> segment_types(nlevels, DB_BLOCKCENT);
                for (int l = 0; l < nlevels; ++l)
                  segment_ids[l] = l;
                ierr = DBAddRegionArray(
                    mrgtree.get(), nlevels, region_name_ptrs.data(), 0,
                    levelmaps_name.c_str(), 1, segment_ids.data(),
                    ncomps_level.data(), segment_types.data(), nullptr);
                assert(!ierr);
                ierr = DBSetCwr(mrgtree.get(), "..");
                assert(ierr >= 0);
              }

              {
                ierr =
                    DBAddRegion(mrgtree.get(), "patches", 0, ncomps_total,
                                nullptr, 0, nullptr, nullptr, nullptr, nullptr);
                assert(ierr >= 0);
                ierr = DBSetCwr(mrgtree.get(), "patches");
                assert(ierr >= 0);
                const std::vector<std::string> region_names{"@patch%d@n"};
                std::vector<const char *> region_name_ptrs;
                region_name_ptrs.reserve(region_names.size());
                for (const auto &n : region_names)
                  region_name_ptrs.push_back(n.c_str());
                std::vector<int> segment_ids(ncomps_total);
                std::vector<int> segment_types(ncomps_total, DB_BLOCKCENT);
                for (int c = 0; c < ncomps_total; ++c)
                  segment_ids[c] = c;
                ierr = DBAddRegionArray(
                    mrgtree.get(), ncomps_total, region_name_ptrs.data(), 0,
                    childmaps_name.c_str(), 1, segment_ids.data(),
                    num_children.data(), segment_types.data(), nullptr);
                ierr = DBSetCwr(mrgtree.get(), "..");
                assert(ierr >= 0);
              }

              {
                const std::vector<std::string> mrgv_onames{
                    multimeshname + "_wmrgtree_lvlRatios",
                    multimeshname + "_wmrgtree_ijkExts",
                    multimeshname + "_wmrgtree_xyzExts", "rank"};
                std::vector<const char *> mrgv_oname_ptrs;
                mrgv_oname_ptrs.reserve(mrgv_onames.size() + 1);
                for (const auto &n : mrgv_onames)
                  mrgv_oname_ptrs.push_back(n.c_str());
                mrgv_oname_ptrs.push_back(nullptr);

                const DB::ptr<DBoptlist> mt_optlist =
                    DB::make(DBMakeOptlist(10));
                assert(mt_optlist);
                ierr = DBAddOption(mt_optlist.get(), DBOPT_MRGV_ONAMES,
                                   mrgv_oname_ptrs.data());
                assert(!ierr);
                ierr =
                    DBPutMrgtree(metafile.get(), mrgtree_name.c_str(),
                                 "amr_mesh", mrgtree.get(), mt_optlist.get());
                assert(!ierr);
              }
            }

            {
              const std::string levelrationame =
                  multimeshname + "_wmrgtree_lvlRatios";
              const std::vector<std::string> compnames{"iRatio", "jRatio"};
              std::vector<const char *> compname_ptrs;
              compname_ptrs.reserve(compnames.size());
              for (const auto &n : compnames)
                compname_ptrs.push_back(n.c_str());
              const std::vector<std::string> regionnames{"@level%d@n"};
              std::vector<const char *> regionname_ptrs;
              regionname_ptrs.reserve(regionnames.size());
              for (const auto &n : regionnames)
                regionname_ptrs.push_back(n.c_str());
              // One ratio value per region (= per level): DBPutMrgvar reads
              // nlevels values from each component pointer, so the buffer must
              // hold nlevels entries (every level refines by 2). A single entry
              // would make Silo read past the end for nlevels > 1.
              std::array<std::vector<int>, 2> ratio_data;
              for (int d = 0; d < 2; ++d)
                ratio_data[d].assign(nlevels, 2);
              std::array<const void *, 2> ratio_data_ptrs;
              for (int d = 0; d < 2; ++d)
                ratio_data_ptrs[d] = ratio_data[d].data();
              ierr = DBPutMrgvar(metafile.get(), levelrationame.c_str(),
                                 mrgtree_name.c_str(), 2, compname_ptrs.data(),
                                 nlevels, regionname_ptrs.data(), DB_INT,
                                 ratio_data_ptrs.data(), nullptr);
              assert(!ierr);
            }

            {
              const std::string iextentsname =
                  multimeshname + "_wmrgtree_ijkExts";
              const std::string extentsname =
                  multimeshname + "_wmrgtree_xyzExts";
              const std::vector<std::string> icompnames{"iMin", "iMax", "jMin",
                                                        "jMax"};
              const char *a_name = (axis_a == 0)   ? "x"
                                   : (axis_a == 1) ? "y"
                                                   : "z";
              const char *b_name = (axis_b == 0)   ? "x"
                                   : (axis_b == 1) ? "y"
                                                   : "z";
              const std::vector<std::string> compnames{
                  std::string(a_name) + "Min", std::string(a_name) + "Max",
                  std::string(b_name) + "Min", std::string(b_name) + "Max"};
              std::vector<const char *> icompname_ptrs;
              std::vector<const char *> compname_ptrs;
              icompname_ptrs.reserve(icompnames.size());
              compname_ptrs.reserve(compnames.size());
              for (const auto &n : icompnames)
                icompname_ptrs.push_back(n.c_str());
              for (const auto &n : compnames)
                compname_ptrs.push_back(n.c_str());
              const std::vector<std::string> regionnames{"@patch%d@n"};
              std::vector<const char *> regionname_ptrs;
              regionname_ptrs.reserve(regionnames.size());
              for (const auto &n : regionnames)
                regionname_ptrs.push_back(n.c_str());

              std::array<std::array<std::vector<int>, 2>, 2> idata;
              std::array<std::array<std::vector<CCTK_REAL>, 2>, 2> rdata;
              for (int d = 0; d < 2; ++d)
                for (int f = 0; f < 2; ++f) {
                  idata[d][f].reserve(ncomps_total);
                  rdata[d][f].reserve(ncomps_total);
                }
              for (const auto &s : slabs)
                for (int d = 0; d < 2; ++d) {
                  idata[d][0].push_back(s.ilo[d]);
                  idata[d][1].push_back(s.ihi[d]);
                  rdata[d][0].push_back(s.xlo[d]);
                  rdata[d][1].push_back(s.xhi[d]);
                }
              std::array<std::array<const void *, 2>, 2> idata_ptrs;
              std::array<std::array<const void *, 2>, 2> rdata_ptrs;
              for (int d = 0; d < 2; ++d)
                for (int f = 0; f < 2; ++f) {
                  idata_ptrs[d][f] = idata[d][f].data();
                  rdata_ptrs[d][f] = rdata[d][f].data();
                }
              ierr = DBPutMrgvar(
                  metafile.get(), iextentsname.c_str(), mrgtree_name.c_str(),
                  2 * 2, icompname_ptrs.data(), ncomps_total,
                  regionname_ptrs.data(), DB_INT, idata_ptrs.data(), nullptr);
              assert(!ierr);
              ierr = DBPutMrgvar(
                  metafile.get(), extentsname.c_str(), mrgtree_name.c_str(),
                  2 * 2, compname_ptrs.data(), ncomps_total,
                  regionname_ptrs.data(), db_datatype_v<CCTK_REAL>,
                  rdata_ptrs.data(), nullptr);
              assert(!ierr);
              const std::vector<int> ranks(ncomps_total, 2);
              const std::vector<const void *> rank_ptrs{ranks.data()};
              ierr =
                  DBPutMrgvar(metafile.get(), "rank", mrgtree_name.c_str(), 1,
                              nullptr, ncomps_total, regionname_ptrs.data(),
                              DB_INT, rank_ptrs.data(), nullptr);
              assert(!ierr);
            }
          } // if (emit_amr)

          std::vector<const char *> meshname_ptrs;
          meshname_ptrs.reserve(meshnames.size());
          for (const auto &s : meshnames)
            meshname_ptrs.push_back(s.c_str());

          const DB::ptr<DBoptlist> optlist = DB::make(DBMakeOptlist(10));
          assert(optlist);
          int cycle = cctk_iteration;
          ierr = DBAddOption(optlist.get(), DBOPT_CYCLE, &cycle);
          assert(!ierr);
          double dtime = cctk_time;
          ierr = DBAddOption(optlist.get(), DBOPT_DTIME, &dtime);
          assert(!ierr);
          int quadmesh = DB_QUADMESH;
          ierr = DBAddOption(optlist.get(), DBOPT_MB_BLOCK_TYPE, &quadmesh);
          assert(!ierr);

          int extents_size = 2 * 2;
          typedef std::array<std::array<double, 2>, 2> dextent_t;
          std::vector<dextent_t> dextents(ncomps_total);
          for (int i = 0; i < ncomps_total; ++i)
            for (int d = 0; d < 2; ++d) {
              dextents[i][0][d] = slabs[i].xlo[d];
              dextents[i][1][d] = slabs[i].xhi[d];
            }
          ierr = DBAddOption(optlist.get(), DBOPT_EXTENTS_SIZE, &extents_size);
          assert(!ierr);
          ierr = DBAddOption(optlist.get(), DBOPT_EXTENTS, dextents.data());
          assert(!ierr);

          std::vector<int> zonecounts(ncomps_total);
          for (int i = 0; i < ncomps_total; ++i)
            zonecounts[i] = slabs[i].zonecount;
          ierr =
              DBAddOption(optlist.get(), DBOPT_ZONECOUNTS, zonecounts.data());
          assert(!ierr);

          // Only point the multimesh at the mrgtree when one was actually
          // written (see emit_amr above); otherwise this is a plain multimesh.
          if (emit_amr) {
            ierr = DBAddOption(optlist.get(), DBOPT_MRGTREE_NAME,
                               const_cast<char *>(mrgtree_name.c_str()));
            assert(!ierr);
          }

          // Nameschemes: write the per-block index arrays, then use the
          // patterns only if their file-based expansion reproduces the
          // explicit names (see nameschemes_match above); otherwise fall
          // back to the explicit list.
          int mesh_ns_nblocks = 0;
          std::string file_ns, mesh_ns; // must outlive DBPutMultimesh
          if (silo_planes_nameschemes) {
            std::vector<int> ns_patch(ncomps_total), ns_level(ncomps_total),
                ns_comp(ncomps_total), ns_file(ncomps_total);
            for (int i = 0; i < ncomps_total; ++i) {
              ns_patch[i] = slabs[i].patch;
              ns_level[i] = slabs[i].level;
              ns_comp[i] = slabs[i].component;
              ns_file[i] = slabs[i].proc / ioproc_every;
            }
            const int adims = ncomps_total;
            ierr =
                DBWrite(metafile.get(), (multimeshname + "_ns_patch").c_str(),
                        ns_patch.data(), &adims, 1, DB_INT);
            assert(!ierr);
            ierr =
                DBWrite(metafile.get(), (multimeshname + "_ns_level").c_str(),
                        ns_level.data(), &adims, 1, DB_INT);
            assert(!ierr);
            ierr = DBWrite(metafile.get(), (multimeshname + "_ns_comp").c_str(),
                           ns_comp.data(), &adims, 1, DB_INT);
            assert(!ierr);
            ierr = DBWrite(metafile.get(), (multimeshname + "_ns_file").c_str(),
                           ns_file.data(), &adims, 1, DB_INT);
            assert(!ierr);

            file_ns = make_file_ns(multimeshname);
            mesh_ns =
                make_block_ns(make_plane_meshname(plane.tag, centering_tag,
                                                  mesh_props.nghosts, 0, 0, 0),
                              multimeshname);
            if (nameschemes_match(metafile.get(), file_ns, mesh_ns,
                                  meshnames)) {
              mesh_ns_nblocks = ncomps_total;
              ierr = DBAddOption(optlist.get(), DBOPT_MB_FILE_NS,
                                 const_cast<char *>(file_ns.c_str()));
              assert(!ierr);
              ierr = DBAddOption(optlist.get(), DBOPT_MB_BLOCK_NS,
                                 const_cast<char *>(mesh_ns.c_str()));
              assert(!ierr);
            } else {
              CCTK_VWARN(CCTK_WARN_ALERT,
                         "OutputSiloPlanes: namescheme expansion does not "
                         "reproduce the explicit block names for multimesh "
                         "%s; writing explicit name lists instead",
                         multimeshname.c_str());
            }
          }

          ierr = DBPutMultimesh(
              metafile.get(), multimeshname.c_str(), ncomps_total,
              mesh_ns_nblocks ? nullptr : meshname_ptrs.data(), nullptr,
              optlist.get());
          assert(!ierr);
          meta_meshes[mesh_props] = mesh_ns_nblocks;
        }

        const std::string multimeshname =
            make_plane_meshname(plane.tag, centering_tag, mesh_props.nghosts);
        // The multivars can reuse the multimesh's index arrays only when this
        // group's block list has the same length as the one the arrays were
        // written for (membership is mirrored across groups by the owner-cell
        // design; the expansion check guards the contents, but indexing an
        // external array out of range is undefined, so the length is checked
        // first). A multimesh written with explicit lists has no arrays.
        const bool try_var_ns = silo_planes_nameschemes &&
                                meta_meshes.at(mesh_props) == ncomps_total;
        // The file namescheme is the same for every variable of the group.
        const std::string vfile_ns =
            try_var_ns ? make_file_ns(multimeshname) : std::string();

        for (int vi = 0; vi < numvars; ++vi) {
          const std::string multivarname =
              make_plane_varname(gi, vi, plane.tag);

          const DB::ptr<DBoptlist> mv_optlist = DB::make(DBMakeOptlist(10));
          assert(mv_optlist);
          int cycle = cctk_iteration;
          ierr = DBAddOption(mv_optlist.get(), DBOPT_CYCLE, &cycle);
          assert(!ierr);
          double dtime = cctk_time;
          ierr = DBAddOption(mv_optlist.get(), DBOPT_DTIME, &dtime);
          assert(!ierr);
          ierr = DBAddOption(mv_optlist.get(), DBOPT_MMESH_NAME,
                             const_cast<char *>(multimeshname.c_str()));
          assert(!ierr);
          int vartype_scalar = DB_VARTYPE_SCALAR;
          ierr =
              DBAddOption(mv_optlist.get(), DBOPT_TENSOR_RANK, &vartype_scalar);
          assert(!ierr);
          int quadvar = DB_QUADVAR;
          ierr = DBAddOption(mv_optlist.get(), DBOPT_MB_BLOCK_TYPE, &quadvar);
          assert(!ierr);

          bool var_ns_ok = false;
          std::string var_ns; // must outlive DBPutMultivar
          if (try_var_ns) {
            var_ns = make_block_ns(
                make_plane_varname(gi, vi, plane.tag, 0, 0, 0), multimeshname);
            var_ns_ok = nameschemes_match(metafile.get(), vfile_ns, var_ns,
                                          varnames_per_var[vi]);
            if (var_ns_ok) {
              ierr = DBAddOption(mv_optlist.get(), DBOPT_MB_FILE_NS,
                                 const_cast<char *>(vfile_ns.c_str()));
              assert(!ierr);
              ierr = DBAddOption(mv_optlist.get(), DBOPT_MB_BLOCK_NS,
                                 const_cast<char *>(var_ns.c_str()));
              assert(!ierr);
            } else {
              CCTK_VWARN(CCTK_WARN_ALERT,
                         "OutputSiloPlanes: namescheme expansion does not "
                         "reproduce the explicit block names for multivar "
                         "%s; writing explicit name lists instead",
                         multivarname.c_str());
            }
          }

          std::vector<const char *> varname_ptrs;
          if (!var_ns_ok) {
            varname_ptrs.reserve(varnames_per_var[vi].size());
            for (const auto &s : varnames_per_var[vi])
              varname_ptrs.push_back(s.c_str());
          }
          ierr =
              DBPutMultivar(metafile.get(), multivarname.c_str(), ncomps_total,
                            var_ns_ok ? nullptr : varname_ptrs.data(), nullptr,
                            mv_optlist.get());
          assert(!ierr);
        }
      }

      {
        const std::string visitname = output_dir + "/" + output_file + "." +
                                      plane.tag + ".silo_planes.visit";
        std::ofstream visit(visitname, std::ios::app);
        assert(visit.good());
        visit << make_plane_filename(output_file, plane.tag, cctk_iteration)
              << "\n";
      }

      {
        output_file_description_t ofd;
        ofd.filename = metafilename;
        ofd.description =
            "2D plane CarpetX HDF5 Silo output (" + plane.tag + ")";
        ofd.writer_thorn = CCTK_THORNSTRING;
        for (int gi = 0; gi < CCTK_NumGroups(); ++gi) {
          if (!output_group.at(gi))
            continue;
          if (CCTK_GroupTypeI(gi) != CCTK_GF)
            continue;
          const int numvars = CCTK_NumVarsInGroupI(gi);
          assert(numvars >= 0);
          const int firstvar = CCTK_FirstVarIndexI(gi);
          for (int vi = 0; vi < numvars; ++vi)
            ofd.variables.push_back(CCTK_FullVarName(firstvar + vi));
        }
        ofd.iterations = {cctk_iteration};
        ofd.output_directions.push_back(axis_a);
        ofd.output_directions.push_back(axis_b);
        ofd.format_name = "CarpetX/Silo/HDF5";
        ofd.format_version = {1, 0, 0};
        OutputMeta_RegisterOutputFile(std::move(ofd));
      }
    }
  }
}

} // namespace PlanesX

#endif // HAVE_CAPABILITY_Silo
