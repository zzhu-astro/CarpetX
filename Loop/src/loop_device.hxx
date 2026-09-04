#ifndef CARPETX_LOOP_LOOP_DEVICE_HXX
#define CARPETX_LOOP_LOOP_DEVICE_HXX

#include "loop.hxx"

#include <cctk.h>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#ifdef AMREX_USE_GPU
#include <AMReX_Gpu.H>
#endif

#include <cassert>
#include <cstdint>

#ifndef HAVE_CAPABILITY_Loop
#error                                                                         \
    "Using #include <loop_device.hxx> requires the capability 'Loop' in the calling thorn. Add this to the thorn's 'configuration.ccl' file."
#endif

namespace Loop {

// The struct GridDescBaseDevice is very similar to GridDescBase, except that it
// requires AMReX include files, and thus "REQUIRES CarpetX" in the using
// thorns' configuration.ccl file.
struct GridDescBaseDevice : GridDescBase {

protected:
  GridDescBaseDevice() : GridDescBase() {}

public:
  GridDescBaseDevice(const GridDescBaseDevice &) = default;
  GridDescBaseDevice(GridDescBaseDevice &&) = default;
  GridDescBaseDevice &operator=(const GridDescBaseDevice &) = default;
  GridDescBaseDevice &operator=(GridDescBaseDevice &&) = default;

  GridDescBaseDevice(const cGH *cctkGH) : GridDescBase(cctkGH) {}

  // Loop over a given box
  //
  // NT is the CUDA/HIP block size, i.e. the max-threads argument of
  // __launch_bounds__. MB is the optional SECOND __launch_bounds__ argument,
  // min-blocks-per-multiprocessor: it tells the compiler that at least MB
  // blocks must be resident per SM/CU, which caps the register budget per
  // thread and so raises achieved occupancy. Use it for kernels that are
  // register-bound (occupancy 1) and latency-bound, where trading a few
  // selective spills for a second resident wave is a win.
  //
  // MB == 0 (the default) means "unspecified" and takes the ordinary
  // amrex::ParallelFor<NT> path, byte for byte as before, so every existing
  // caller -- including the loop_*_device wrappers below, which do not
  // forward MB -- is completely unaffected.
  //
  // MB is a CODEGEN DIRECTIVE ONLY. It changes register allocation and
  // spilling, never the arithmetic, the iteration space, the traversal order
  // or the results; a kernel launched with MB != 0 must produce bit-identical
  // output to the same kernel with MB == 0.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, int MB = 0, typename F>
  void loop_box_device(const vect<int, dim> &restrict bnd_min,
                       const vect<int, dim> &restrict bnd_max,
                       const vect<int, dim> &restrict loop_min,
                       const vect<int, dim> &restrict loop_max,
                       const F &f) const {
#ifndef AMREX_USE_GPU

    return this->template loop_box<CI, CJ, CK, VS, N>(bnd_min, bnd_max,
                                                      loop_min, loop_max, f);

#else
    // Run on GPU

    static_assert(CI == 0 || CI == 1);
    static_assert(CJ == 0 || CJ == 1);
    static_assert(CK == 0 || CK == 1);
    static_assert(VS > 0);
    static_assert(N >= 0);
    static_assert(NT > 0);

    static_assert(MB >= 0);

    static_assert(VS == 1, "Only vector size 1 is supported on GPUs");

    if (N == 0 || any(loop_max <= loop_min))
      return;

    // For some reason, the arguments loop_min and loop_max cannot be captured
    // correctly in CUDA, but copies of them can
    const auto bnd_min1 = bnd_min;
    const auto bnd_max1 = bnd_max;
    const auto loop_min1 = loop_min;
    const auto loop_max1 = loop_max;

    // Convert to AMReX box
    const amrex::Box box(
        amrex::IntVect(loop_min[0], loop_min[1], loop_min[2]),
        amrex::IntVect(loop_max[0] - 1, loop_max[1] - 1, loop_max[2] - 1),
        amrex::IntVect(CI ? amrex::IndexType::CELL : amrex::IndexType::NODE,
                       CJ ? amrex::IndexType::CELL : amrex::IndexType::NODE,
                       CK ? amrex::IndexType::CELL : amrex::IndexType::NODE));

    const auto kernel =
        [=, *this] CCTK_DEVICE(const int i, const int j,
                               const int k) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          const vect<int, dim> I = {i, j, k};
          const vect<int, dim> NI =
              vect<int, dim>(I > bnd_max1 - 1) - vect<int, dim>(I < bnd_min1);
          const vect<int, dim> I0 =
              // If NI<0,   return bnd_min1
              // If NI>0,   return bnd_max1-1
              // If NI==0,  return 0
              if_else(NI == 0, 0, if_else(NI < 0, bnd_min1, bnd_max1 - 1));
          const vect<int, dim> BI =
              vect<int, dim>(I == bnd_max1 - 1) - vect<int, dim>(I == bnd_min1);
          for (int iter = 0; iter < N; ++iter) {
            const PointDesc p =
                point_desc({CI, CJ, CK}, I, iter, NI, I0, BI, bnd_min1,
                           bnd_max1, loop_min1, loop_max1);
            f(p);
          }
        };

    if constexpr (MB == 0) {

      amrex::ParallelFor<NT>(box, kernel);

    } else {
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

      // amrex::ParallelFor<NT>(box, ·) launches through AMREX_LAUNCH_KERNEL,
      // which only ever instantiates launch_global<NT> -- max threads, no
      // min-blocks (AMReX_GpuLaunch.H). AMReX does provide the three-argument
      // launch_global<max_threads, min_blocks, L> (AMReX_GpuLaunchGlobal.H),
      // it is simply not reachable through ParallelFor. So mirror the body of
      // ParallelFor<MT>(BoxND, L) here (AMReX_GpuLaunchFunctsG.H) and launch
      // through that overload instead.
      //
      // Keep this in sync with AMReX if the ParallelFor internals change. Note
      // it must call detail::call_f_intvect_handler rather than the lambda
      // directly: the callable built above takes (i, j, k), and the handler is
      // what adapts an IntVect to that signature (and supplies the optional
      // Gpu::Handler argument for callables that want it).
      const amrex::BoxIndexer indexer(box);
      const auto &nec = amrex::Gpu::makeNExecutionConfigs<NT>(box);
      for (const auto &ec : nec) {
        const auto start_idx = std::uint64_t(ec.start_idx);
        const auto kernel1 = [=] AMREX_GPU_DEVICE() noexcept {
          const auto icell =
              std::uint64_t(NT) * blockIdx.x + threadIdx.x + start_idx;
          if (icell < indexer.numPts()) {
            const auto iv = indexer.intVect(icell);
            amrex::detail::call_f_intvect_handler(
                kernel, iv,
                amrex::Gpu::Handler(
                    amrex::min((indexer.numPts() - icell +
                                std::uint64_t(threadIdx.x)),
                               std::uint64_t(NT))));
          }
        };
#if defined(AMREX_USE_CUDA)
        amrex::launch_global<NT, MB>
            <<<ec.nblocks, NT, 0, amrex::Gpu::gpuStream()>>>(kernel1);
#else
        // HIP_KERNEL_NAME protects the comma in the template argument list
        // from hipLaunchKernelGGL's macro argument splitting.
        hipLaunchKernelGGL(HIP_KERNEL_NAME(amrex::launch_global<NT, MB>),
                           ec.nblocks, NT, 0, amrex::Gpu::gpuStream(), kernel1);
#endif
      }
      AMREX_GPU_ERROR_CHECK();

#else

      // Other GPU backends (SYCL) have no launch_global and no
      // min-blocks-per-multiprocessor equivalent. MB is a codegen hint, so
      // dropping it changes performance only, never results.
      amrex::ParallelFor<NT>(box, kernel);

#endif
    }

    static const bool gpu_sync_after_every_kernel = []() {
      int type;
      const void *const ptr =
          CCTK_ParameterGet("gpu_sync_after_every_kernel", "CarpetX", &type);
      assert(ptr);
      assert(type == PARAMETER_BOOLEAN);
      return *static_cast<const CCTK_INT *>(ptr);
    }();
    if (gpu_sync_after_every_kernel) {
      amrex::Gpu::synchronize();
      AMREX_GPU_ERROR_CHECK();
    }
#endif
  }

  // Loop over all points
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_all_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin, imax;
    box_all<CI, CJ, CK>(group_nghostzones, imin, imax);
    loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin, imax, f);
  }

  // Loop over all interior points
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_int_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin, imax;
    box_int<CI, CJ, CK>(group_nghostzones, imin, imax);
    loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin, imax, f);
  }

  // Loop over all points in cell-centered direction, and interior points in
  // vertex-centered direction
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_mix_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin_int, imax_int;
    vect<int, dim> imin_all, imax_all;
    box_int<CI, CJ, CK>(group_nghostzones, imin_int, imax_int);
    box_all<CI, CJ, CK>(group_nghostzones, imin_all, imax_all);

    constexpr vect<int, dim> facetype{CI, CJ, CK};

    vect<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      imin[d] = facetype[d] ? imin_all[d] : imin_int[d];
      imax[d] = facetype[d] ? imax_all[d] : imax_int[d];
    }
    loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin, imax, f);
  }

  // Loop over all points in cell-centered direction, and interior points
  // plus n in vertex-centered direction
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_mixpn_device(const vect<int, dim> &group_nghostzones, const int ord, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin_int, imax_int;
    vect<int, dim> imin_all, imax_all;
    box_int<CI, CJ, CK>(group_nghostzones, imin_int, imax_int);
    box_all<CI, CJ, CK>(group_nghostzones, imin_all, imax_all);

    constexpr vect<int, dim> facetype{CI, CJ, CK};

    vect<int, dim> imin, imax;
    for (int d = 0; d < dim; ++d) {
      imin[d] = facetype[d] ? imin_all[d] : imin_int[d] - ord;
      imax[d] = facetype[d] ? imax_all[d] : imax_int[d] + ord;
    }
    loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin, imax, f);
  }

  // Loop over all points, excluding the outermost point
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_allm1_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin, imax;
    box_all<CI, CJ, CK>(group_nghostzones, imin, imax);
    loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin + 1, imax - 1,
                                           f);
  }

  // Loop over all points excluding outer n points
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_allmn_device(const vect<int, dim> &group_nghostzones, const int ord, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> imin, imax;
    box_all<CI, CJ, CK>(group_nghostzones, imin, imax);
    loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin + ord, imax - ord,
                                           f);
  }

  // Loop over n outer (ghost OR boundary) points. 
  // Loop over faces first, then edges, then corners.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_outer_n_device(const vect<int, dim> &group_nghostzones,
                     const int ord, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              const vect<int, dim> inormal{ni, nj, nk};

              vect<int, dim> imin, imax;
              for (int d = 0; d < dim; ++d) {
                switch (inormal[d]) {
                case -1: // lower boundary
                  imin[d] = all_min[d];
                  imax[d] = all_min[d] + ord;
                  break;
                case 0: // interior
                  imin[d] = int_min[d];
                  imax[d] = int_max[d];
                  break;
                case +1: // upper boundary
                  imin[d] = all_max[d] - ord;
                  imax[d] = all_max[d];
                  break;
                default:
                  assert(0);
                }

                using std::min, std::max;
                imin[d] = max(tmin[d], imin[d]);
                imax[d] = min(tmax[d], imax[d]);
              }

              loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                     imax, f);
            } // if rank
          }
        }
      }
    } // for rank
  }

  // Loop over a part of the domain. Loop over the interior first,
  // then faces, then edges, then corners.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_there_device(const vect<int, dim> &group_nghostzones,
                    const vect<vect<vect<bool, dim>, dim>, dim> &there,
                    const F &f) const {
    constexpr vect<int, dim> offset{CI, CJ, CK};

    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if (there[ni + 1][nj + 1][nk + 1]) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0: // interior
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1: // upper boundary
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::max, std::min;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                       imax, f);
              }
            } // if rank
          }
        }
      }

    } // for rank
  }

  // Loop over all outer boundary points. This excludes ghost faces, but
  // includes ghost edges/corners on non-ghost faces. Loop over faces first,
  // then edges, then corners.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_bnd_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni != 0 && bbox[ni < 0 ? 0 : 1][0]) ||
                  (nj != 0 && bbox[nj < 0 ? 0 : 1][1]) ||
                  (nk != 0 && bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0: // interior
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1: // upper boundary
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                       imax, f);
              }
            } // if rank
          }
        }
      }

    } // for rank
  }

  // Loop over the outermost "boundary" points in the interior. They correspond
  // to points that are shifted inwards by = cctk_nghostzones[3] from those that
  // CarpetX identifies as boundary points. From the perspective of CarpetX (or
  // AMReX), these do not belong in the outer boundary, but rather the interior.
  // This excludes ghost faces, but includes ghost edges/corners on non-ghost
  // faces. Loop over faces first, then edges, then corners. Modified from
  // loop_bnd_device. The additional argument is_sym_bnd is used to indicate
  // whether each boundary is a symmetry boundary. This symmetry information can
  // be provided by the PatchData object in the CarpetX driver.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_outermost_int_device(const vect<int, dim> &group_nghostzones,
                            const vect<vect<bool, dim>, 2> &is_sym_bnd,
                            const F &f) const {
    // boundary_box sets bnd_min and bnd_max
    vect<int, dim> bnd_min, bnd_max;
    // if on (Carpetx) boundary points, then
    //   bnd_min = nghostzones - lbnd
    //   bnd_max = gsh - offset - nghostzones - lbnd
    //   where lbnd : an array of cctk dim integers containing the lowest index
    //   (in each direction) of the local grid, as seen on the global grid
    // else,
    //   bnd_min = -100000000
    //   bnd_max = gsh + 1000000
    // where cctk_gsh : an array of cctk dim integers with the global grid size
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);

    // domain_boxes sets all_min, all_max, int_min and int_max
    vect<int, dim> all_min, all_max, int_min, int_max;
    // Indices for the entire grid
    //   all_min = ghost_offset;
    //   all_max = lsh - offset - ghost_offset;
    // Indices for the interior
    //   int_min = nghostzones;
    //   int_max = lsh - offset - nghostzones;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    // Check that the actual number of ghost zones isn't overridden
    for (int d = 0; d < dim; ++d) {
      assert(group_nghostzones[d] == nghostzones[d]);
    }

    // Shift the indices towards the interior by nghostzones
    all_min += nghostzones;
    int_min += nghostzones;
    bnd_min += nghostzones;
    all_max -= nghostzones;
    int_max -= nghostzones;
    bnd_max -= nghostzones;

    // rank selects: faces, edges, corners
    // rank(interior) = 3
    // rank(face)     = 2
    // rank(edge)     = 1
    // rank(corner)   = 0
    for (int rank = dim - 1; rank >= 0; --rank) {

      // Nested loops that determine {ni,nj,nk} components of the normal vector
      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            // True when the normal vector is normal to a {face, edge, corner}
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {
              // True when point is on left/right boundary,
              // and vector is not parallel to a {face,corner,edge}
              // In either of the 3 directions
              if ((ni != 0 && bbox[ni < 0 ? 0 : 1][0] &&
                   !is_sym_bnd[ni < 0 ? 0 : 1][0]) ||
                  (nj != 0 && bbox[nj < 0 ? 0 : 1][1] &&
                   !is_sym_bnd[nj < 0 ? 0 : 1][1]) ||
                  (nk != 0 && bbox[nk < 0 ? 0 : 1][2] &&
                   !is_sym_bnd[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk}; // normal vector

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0: // interior
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1: // upper boundary
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  // Adjust loop_box index range according to tile min, max
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                       imax, f);
              }
            } // if rank
          }
        }
      }

    } // for rank
  }

#if 0
  // Loop over all outer ghost points. This includes ghost edges/corners on
  // non-ghost faces. Loop over faces first, then edges, then corners.
  template <int CI, int CJ, int CK, int N = 1, int VS = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_ghosts_inclusive_device(const vect<int, dim> &group_nghostzones,
                               const F &f) const {
    constexpr vect<int, dim> offset{CI, CJ, CK};

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni != 0 && !bbox[ni < 0 ? 0 : 1][0]) ||
                  (nj != 0 && !bbox[nj < 0 ? 0 : 1][1]) ||
                  (nk != 0 && !bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  const int ghost_offset =
                      nghostzones[d] - group_nghostzones[d];
                  const int begin_bnd = ghost_offset;
                  const int begin_int = nghostzones[d];
                  const int end_int = lsh[d] - offset[d] - nghostzones[d];
                  const int end_bnd = lsh[d] - offset[d] - ghost_offset;
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = begin_bnd;
                    imax[d] = begin_int;
                    break;
                  case 0: // interior
                    imin[d] = begin_int;
                    imax[d] = end_int;
                    break;
                  case +1: // upper boundary
                    imin[d] = end_int;
                    imax[d] = end_bnd;
                    break;
                  default:
                    assert(0);
                  }

                  imin[d] = std::max(tmin[d], imin[d]);
                  imax[d] = std::min(tmax[d], imax[d]);
                }

                loop_box_boundary_device<CI, CJ, CK, VS, N, NT>(imin, imax,
                                                                inormal, f);
              }
            } // if rank
          }
        }
      }
    } // for rank
  }
#endif

  // Loop over all outer ghost points. This excludes ghost edges/corners on
  // non-ghost faces. Loop over faces first, then edges, then corners.
  template <int CI, int CJ, int CK, int VS = 1, int N = 1,
            int NT = AMREX_GPU_MAX_THREADS, typename F>
  inline CCTK_KERNEL void
  loop_ghosts_device(const vect<int, dim> &group_nghostzones,
                     const F &f) const {
    vect<int, dim> bnd_min, bnd_max;
    boundary_box<CI, CJ, CK>(group_nghostzones, bnd_min, bnd_max);
    vect<int, dim> all_min, all_max, int_min, int_max;
    domain_boxes<CI, CJ, CK>(group_nghostzones, all_min, all_max, int_min,
                             int_max);

    for (int rank = dim - 1; rank >= 0; --rank) {

      for (int nk = -1; nk <= +1; ++nk) {
        for (int nj = -1; nj <= +1; ++nj) {
          for (int ni = -1; ni <= +1; ++ni) {
            if ((ni == 0) + (nj == 0) + (nk == 0) == rank) {

              if ((ni == 0 || !bbox[ni < 0 ? 0 : 1][0]) &&
                  (nj == 0 || !bbox[nj < 0 ? 0 : 1][1]) &&
                  (nk == 0 || !bbox[nk < 0 ? 0 : 1][2])) {

                const vect<int, dim> inormal{ni, nj, nk};

                vect<int, dim> imin, imax;
                for (int d = 0; d < dim; ++d) {
                  switch (inormal[d]) {
                  case -1: // lower boundary
                    imin[d] = all_min[d];
                    imax[d] = int_min[d];
                    break;
                  case 0: // interior
                    imin[d] = int_min[d];
                    imax[d] = int_max[d];
                    break;
                  case +1: // upper boundary
                    imin[d] = int_max[d];
                    imax[d] = all_max[d];
                    break;
                  default:
                    assert(0);
                  }

                  using std::min, std::max;
                  imin[d] = max(tmin[d], imin[d]);
                  imax[d] = min(tmax[d], imax[d]);
                }

                loop_box_device<CI, CJ, CK, VS, N, NT>(bnd_min, bnd_max, imin,
                                                       imax, f);
              }
            } // if rank
          }
        }
      }
    } // for rank
  }

  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL std::enable_if_t<(where == where_t::everywhere), void>
  loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_all_device<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL std::enable_if_t<(where == where_t::interior), void>
  loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_int_device<CI, CJ, CK>(group_nghostzones, f);
  }
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL std::enable_if_t<(where == where_t::boundary), void>
  loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_bnd_device<CI, CJ, CK>(group_nghostzones, f);
  }
#if 0
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL
      std::enable_if_t<(where == where_t::ghosts_inclusive), void>
      loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_ghosts_inclusive_device<CI, CJ, CK>(group_nghostzones, f);
  }
#endif
  template <int CI, int CJ, int CK, where_t where, typename F>
  inline CCTK_KERNEL std::enable_if_t<(where == where_t::ghosts), void>
  loop_device(const vect<int, dim> &group_nghostzones, const F &f) const {
    loop_ghosts_device<CI, CJ, CK>(group_nghostzones, f);
  }

  template <typename F>
  inline CCTK_KERNEL void
  loop_there_device_idx(const vect<int, dim> &indextype,
                        const vect<vect<vect<bool, dim>, dim>, dim> &there,
                        const vect<int, dim> &group_nghostzones,
                        const F &f) const {
    switch (indextype[0] + 2 * indextype[1] + 4 * indextype[2]) {
    case 0b000:
      return noinline([&] {
        return loop_there_device<0, 0, 0>(group_nghostzones, there, f);
      });
    case 0b001:
      return noinline([&] {
        return loop_there_device<1, 0, 0>(group_nghostzones, there, f);
      });
    case 0b010:
      return noinline([&] {
        return loop_there_device<0, 1, 0>(group_nghostzones, there, f);
      });
    case 0b011:
      return noinline([&] {
        return loop_there_device<1, 1, 0>(group_nghostzones, there, f);
      });
    case 0b100:
      return noinline([&] {
        return loop_there_device<0, 0, 1>(group_nghostzones, there, f);
      });
    case 0b101:
      return noinline([&] {
        return loop_there_device<1, 0, 1>(group_nghostzones, there, f);
      });
    case 0b110:
      return noinline([&] {
        return loop_there_device<0, 1, 1>(group_nghostzones, there, f);
      });
    case 0b111:
      return noinline([&] {
        return loop_there_device<1, 1, 1>(group_nghostzones, there, f);
      });
    default:
      assert(0);
    }
  }

  template <where_t where, typename F>
  inline CCTK_KERNEL void
  loop_device_idx(const vect<int, dim> &indextype,
                  const vect<int, dim> &group_nghostzones, const F &f) const {
    switch (indextype[0] + 2 * indextype[1] + 4 * indextype[2]) {
    case 0b000:
      return noinline(
          [&] { return loop_device<0, 0, 0, where>(group_nghostzones, f); });
    case 0b001:
      return noinline(
          [&] { return loop_device<1, 0, 0, where>(group_nghostzones, f); });
    case 0b010:
      return noinline(
          [&] { return loop_device<0, 1, 0, where>(group_nghostzones, f); });
    case 0b011:
      return noinline(
          [&] { return loop_device<1, 1, 0, where>(group_nghostzones, f); });
    case 0b100:
      return noinline(
          [&] { return loop_device<0, 0, 1, where>(group_nghostzones, f); });
    case 0b101:
      return noinline(
          [&] { return loop_device<1, 0, 1, where>(group_nghostzones, f); });
    case 0b110:
      return noinline(
          [&] { return loop_device<0, 1, 1, where>(group_nghostzones, f); });
    case 0b111:
      return noinline(
          [&] { return loop_device<1, 1, 1, where>(group_nghostzones, f); });
    default:
      assert(0);
    }
  }
};

template <where_t where, typename F>
inline CCTK_KERNEL void
loop_device_idx(const cGH *cctkGH, const vect<int, dim> &indextype,
                const vect<int, dim> &nghostzones, const F &f) {
  GridDescBaseDevice(cctkGH).loop_device_idx<where>(indextype, nghostzones, f);
}

} // namespace Loop

#undef CCTK_CENTERING_GRID
#define CCTK_CENTERING_GRID const Loop::GridDescBaseDevice cctk_grid(cctkGH)

#endif // #ifndef CARPETX_LOOP_LOOP_DEVICE_HXX
