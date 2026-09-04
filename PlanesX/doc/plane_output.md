# 2D Plane Output for Silo and openPMD (thorn PlanesX)

Axis-aligned 2D slabs (xy / xz / yz) at chosen world-coordinate elevations,
written through the Silo and openPMD code paths. File-per-plane, AMR-aware
(per-level snap), Cartesian patches only. Implemented in
`PlanesX/src/planes.{hxx,cxx}` (spec parsing, tag formatting, per-level
snap, 3D-FAB slab extraction), `silo_planes.{hxx,cxx}` (`OutputSiloPlanes`),
and `openpmd_planes.{hxx,cxx}` (`OutputOpenPMDPlanes`); dispatched from
`output.cxx::PlanesX_OutputGH`, an IO method registered at `STARTUP` via
`CCTK_RegisterIOMethod`.

PlanesX was extracted from CarpetX (it originally lived in
`CarpetX/src/io_planes.*` and was dispatched from `io.cxx::OutputGH`).
CarpetX's `OutputGH` overloads `CCTK_OutputGH`; it now ends by traversing the
IO methods registered via `CCTK_RegisterIOMethod` (CarpetX commit "call
registered IO methods from OutputGH" — **required**: without it the callback
is never invoked and no planes are written; `PlanesX_CheckOutputCalled`,
scheduled at `CCTK_TERMINATE`, aborts with a clear error when plane output
was requested but the method never ran). That restores the exact
pre-extraction slot: after the `CCTK_POSTSTEP` *and* `CCTK_ANALYSIS`
traversals — so grid functions computed in the analysis bin (energies,
constraints, errors) are up to date when the planes are written — on every
iteration including recovery (where `CCTK_ANALYSIS` itself is skipped), and
before CarpetX's `OutputMeta`, so files registered via
`OutputMeta_RegisterOutputFile` are described in the same iteration's
metadata. (An earlier revision hooked in at `CCTK_POSTSTEP`, which runs
*before* the analysis bin and would have output analysis-computed grid
functions one iteration stale.) The writers read CarpetX's grid hierarchy
directly (`ghext` via the
exported `driver.hxx`), and follow CarpetX's private output parameters
(`out_mode`, `out_proc_every`, `out_silo_compression_options`,
`openpmd_format`, cadence fallbacks) via `CCTK_ParameterGet`
(`carpetx_params.hxx`); the small openPMD conventions (format mapping, ADIOS2
options, iteration encoding, units, mesh/component naming) are mirrored from
`io_openpmd.cxx` in `openpmd_planes.cxx`.

## Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `PlanesX::silo_planes` | `""` | Specs, e.g. `"xy:0.0, xy:10.0, xz:-5.0"` |
| `PlanesX::silo_plane_vars` | `""` | Variable regex |
| `PlanesX::silo_planes_every` | `-1` | Cadence (-1 ⇒ `CarpetX::out_silo_every` ⇒ `IO::out_every`) |
| `PlanesX::openpmd_planes` / `_plane_vars` / `_planes_every` | — | Same, for openPMD |
| `PlanesX::planes_int_precision` | `4` | Min integer digits in tag |
| `PlanesX::planes_frac_precision` | `3` | Fractional digits in tag |
| `PlanesX::silo_planes_ghosts` | `no` | interior-only Silo slabs (smaller files; VisIt may show box seams); `yes`: ghost-extended slabs |
| `PlanesX::silo_planes_single_precision` | `default` | `yes`/`no`/`default` (= follow `IO::out_single_precision`); Silo data as `DB_FLOAT`. openPMD planes are always `CCTK_REAL` |
| `PlanesX::silo_planes_nameschemes` | `yes` | metafiles use Silo nameschemes (one pattern + small index arrays) instead of one explicit string per block and variable — metafile size independent of block count; see *Nameschemes* below |

Spec syntax: `<axes>:<elevation>`, comma-separated; `<elevation>` is the
world coordinate along the normal axis.

## Snapping & tags

Elevations are rounded to `frac_precision` digits at parse (one-time warning on
overflow); the rounded value is both the tag and the snap target. Per level,
snap to the nearest vertex (VC: `x0 + i·dx`) or cell centre (CC: `x0 +
(i+½)·dx`). A plane outside every Cartesian (patch, level) along its normal —
or a non-Cartesian patch — is skipped with a one-time warning and **no file is
written**.

Tag: `<plane>_<axis>_<sign><int>p<frac>`, e.g. `xy_z_pos0012p500` (z=+12.5),
`xz_y_neg0003p000` (y=−3). Survives Silo `legalize_name` and HDF5/ADIOS2 rules.
Silo mesh names get a 2-char in-plane centering suffix (`cv`/`vc`) for rank-1
staggered groups (per-group mesh); rank-0/2 groups share a mesh (no suffix).

**Recorded true location.** Because the snap is per (level, normal-axis
centering), variables vertex- vs cell-centred along the normal are written at
planes up to `dx/2` apart even within one tagged file. The writers therefore
record the *actual* snapped world coordinate so readers never need to replay
the snap rule: openPMD attaches per-mesh attributes `planeNormalAxis`,
`planeIndex` (the snapped grid index) and `planeCoordinate` (the world
coordinate of that mesh's slab; the iteration-level `planeElevation` remains
the requested, rounded elevation); Silo writes, into every per-plane file
(leaf and metafile), `plane_normal_axis`, `plane_elevation`, and per Cartesian
patch the arrays `plane_ncoord_vertex_m%04d` / `plane_ncoord_cell_m%04d`
(`double[nlevels]`, indexed by level, NaN where the plane misses the level) —
pick the array matching your variable's normal-axis centering.
`verify_planes.py` cross-checks the recorded values against its independent
snap replay.

## Output files

```
Silo:
  <out_dir>/<sim>.<tag>.it<N>.silo_planes.dir/<sim>.<tag>.it<N>.p<NNNNNN>.silo
  <out_dir>/<sim>.<tag>.it<N>.silo            metafile (multimesh/var + DBmrgtree)
  <out_dir>/<sim>.<tag>.silo_planes.visit     VisIt index
openPMD:
  <out_dir>/<sim>.<tag>.it<N>.<ext>           one file per iteration (.bp5/.h5/…)
  <out_dir>/<sim>.<tag>.openpmd.visit         VisIt index
```

- **Silo** writes interior-only slabs by default; `silo_planes_ghosts = yes`
  includes ghost cells (`DBOPT_LO/HI_OFFSET` markers; with 3 ghost zones the
  interior-only saving is ~29% at 32² boxes and ~67% at 8² boxes). Data is
  `CCTK_REAL` or, with
  `silo_planes_single_precision` (or `IO::out_single_precision`), `DB_FLOAT`
  (2× smaller; mesh coordinates and the recorded plane location stay double).
  The full `AllParameters` dump lives in the per-iteration **metafile only**
  (the leaf files of the same iteration would just duplicate it; a steered
  parameter still shows up, since every output iteration writes a fresh
  metafile); the metafile
  carries an AMR `DBmrgtree` (`amr_decomp/levels/patches` + `lvlRatios`/
  `ijkExts`/`xyzExts`/`rank` mrgvars) so VisIt shadows coarse data with finer
  levels where they overlap. The mrgtree (and the multimesh's
  `DBOPT_MRGTREE_NAME`) is written **only when at least one coarse→fine
  parent-child relationship exists** (`emit_amr` in `silo_planes.cxx`);
  single-level output, or a plane that crosses only one level, is a plain
  multimesh — see *mrgtree pitfalls* below.
- **openPMD** is interior-only (ghosts stripped), MPI-collective `storeChunk`,
  one 2D mesh per (group, patch, level); cell-centring via per-component
  `setPosition`. Each iteration also carries AMR-hierarchy attributes
  (`numPatches`, `patchSuffixes`, per-(patch,level) `chunkInfo`/`iteration_*`)
  and plane keys `planeTag`/`planeNormalAxis`/`planeElevation`. Empty
  `chunkInfo` is omitted (a zero-length attribute is unreadable by openpmd-api).
  A `(patch, level, group)` whose boxes don't reach the plane is skipped
  entirely (no mesh record): `snap_to_grid_index` only checks the level's full
  refined domain, so a finer level can pass it without actually covering the
  plane, and defining a mesh with no stored chunk makes openpmd-api emit "No
  extent found" read warnings. The skip uses the replicated BoxArray so all
  ranks agree (mesh creation is collective).

  **Cell-centred dataset extent.** A mesh's extent is the *vertex* count
  (`ncells+1`) per in-plane axis for every centering (same `+1+1` as the 3D
  writer); a cell-centred variable writes only `ncells`, leaving the high-edge
  index as backend fill. Harmless internally — writer `storeChunk` and the
  readers (`available_chunks()`; the 3D recovery reader's `count = box.shape()`)
  address data per-box, never the fill index — but an external full-extent
  reader would see a one-cell fill stripe. Sizing per centering would fix it but
  changes every dataset shape (golden + checkpoints), so it is left as is. Silo
  has no analogue (a zone-centred quadvar is `N` zones on an `N+1`-node mesh).
  **Backend caveat:** ADIOS2's `available_chunks()` returns exactly the written
  boxes, but HDF5 has no chunk bookkeeping and reports the *full extent* as one
  chunk — an HDF5 reader therefore must clip the high-edge fill index off
  cell-centred axes (`plane_readers/openpmd_reader.py` does). For AMR planes
  read from HDF5, the full-extent chunk would additionally contain fill values
  wherever a finer level's boxes do not reach; a fully general HDF5 reader
  should reconstruct the written boxes from the per-level `chunkInfo`
  attributes instead (the `.h5` test coverage, `planes-options.par`, is
  single-level).

## Viewing in VisIt

VisIt reads **Silo** natively: open the `.silo_planes.visit` index and
pseudocolor — the mrgtree gives correct AMR level shadowing. Stock VisIt
(≥ ~3.1) also ships an openPMD reader (`src/databases/OpenPMD/`, the former
`openPMD-visit-plugin`), but it reads **HDF5 only** (raw HDF5 calls, no
openpmd-api, no ADIOS2/BP5), was written for PIC-style files, and does no AMR
level shadowing — so it can at best display planes written with
`CarpetX::openpmd_format = "HDF5"` (nothing in the writer prevents `.h5`, as
long as the linked openPMD-api has the HDF5 backend). ParaView ≥ 5.9 ships an
openpmd-api-based reader that reads the default `.bp5` directly. For VisIt
with AMR shadowing, Silo remains the reliable path.

### mrgtree pitfalls (empty child map; 3D name mismatch)

The Silo AMR `DBmrgtree` has two sharp edges, both found while debugging a VisIt
`SIGSEGV` (`avtSiloFileFormat::GetMesh` → `DBFreeGroupelmap` → `cfree`) on
single-level plane output.

1. **All-empty child map crashes VisIt.** The child map (`_wmrgtree_chldMaps`,
   a `DBPutGroupelmap`) has one segment per component listing that component's
   finer-level children. With only one level — or a plane that intersects a
   single level — *every* segment is empty. Silo then stores no segment-data
   array; on read-back `segment_data == NULL` while `num_segments > 0`, so VisIt
   dereferences a NULL array when freeing the map and crashes. The numeric test
   never sees this: `plane_readers/silo_reader.py` reads the leaf per-box
   `.silo` files directly and never opens the metafile multimesh or walks the
   mrgtree, so CI stays green while VisIt dies. **Fix applied:** the plane writer
   emits the mrgtree only when `total_children > 0`, else a plain multimesh
   (`emit_amr`, `silo_planes.cxx`).

2. **CarpetX's 3D writer (`io_silo.cxx`) hides the same hazard behind a bug.**
   It names the mrgtree object `"mrgTree"` (`DBPutMrgtree`) but sets the
   multimesh's `DBOPT_MRGTREE_NAME` to `"mrgtree"` — a case mismatch. Silo/HDF5
   names are case-sensitive, so VisIt's `DBGetMrgtree` lookup fails, the tree is
   silently ignored, and the multimesh is read as plain. That is why
   single-level 3D Silo output opens fine while the (correctly-wired) plane
   output crashed. The cost is that **3D AMR Silo output has never actually
   received the intended mrgtree level-shadowing in VisIt** — the feature is
   dead code on the read side.

   *Future fix:* align the two names in `io_silo.cxx` so VisIt loads the 3D
   mrgtree — but this **must** land together with the same `emit_amr`/`nlevels >
   1` guard as the plane writer, otherwise single-level 3D output will start
   crashing exactly as the planes did. Consider factoring the shared mrgtree
   construction (group maps, region arrays, ratio/extent mrgvars, the
   `total_children > 0` guard) out of both writers so the two paths cannot drift
   again. A cheaper CI safety net: have the verifier also open the metafile
   multimesh (not just the leaf files) so a tree VisIt would reject fails the
   test.

### FIXED: empty leaf files (`silo_planes_ghosts = no` unreadable in VisIt)

Observed 2026-06-05 on a many-rank Frontier production run (multi-level AMR,
`silo_planes_ghosts = no` + `silo_planes_single_precision = yes`, VisIt 3.4
client/server): a leaf file failed VisIt's Silo reader at *open* time ("It
may be an invalid file"), and the metafile multimesh rendered all white; the
same run with `silo_planes_ghosts = yes` was fine.

**Root cause (reproduced locally with `planes-amr-noghosts.par` on 4 ranks):**
`OutputSiloPlanes` used to `DBCreate` a leaf file on every IO rank for every
in-domain plane *before* knowing whether the rank emits any slab. A rank
owning no intersecting box left a file holding only the plane-location
arrays and no mesh/var object — VisIt's Silo plugin reports such a file as
invalid at open time. With ghosts=no the single-owner membership makes such
ranks routine (a plane crossing only the coarse level has just one box
column of owners); with ghosts they occur on any run large enough that some
rank's boxes all miss the plane, so the ghost path was exposed too, merely
harder to hit at test scale.

**Fix:** the leaf pass creates the file lazily on the first slab the rank
actually writes (`ensure_file` in `silo_planes.cxx`); ranks without a slab
create nothing. The metafile references only files of slab-owning ranks, so
the never-created files were never referenced — directory contents now match
the references exactly. *Note:* this makes the agreement of the two passes'
independently re-derived slab membership (leaf pass per-rank loop vs metafile
`slabs` gather) load-bearing; today they mirror each other line-for-line
(`owner_cell` / `slice_idx` / databox tests) and the CI VisIt gate would catch
a drift, but a future refactor should compute one shared slab list and drive
both passes from it (same factoring argument as for the mrgtree construction
above).

Why the test suite never caught it: `planes-options.par` — previously the
only ghosts=no coverage — is single-level on 2 ranks (every rank owns a
slab), the numeric verifier reads leaf HDF5 components directly (h5py, no
`DBOpen`), and the old `scripts/visit-check-planes.py` opened only the
*first* leaf per directory and was not in CI. All three holes are closed:
`planes-amr-noghosts.par` runs 3-level AMR ghosts=no on 4 ranks, the checker
opens **every** leaf and fails zero-zone/non-finite draws (the all-white
mode), and CI gates on a headless VisIt run (see ci.yml).

## Testing (thorn `TestPlanesX`)

`TestPlanesX` is a rigorous, CI-integrated test suite. It declares one grid
function per index-type centering (the eight V/C combinations) and fills each,
on the interior, with the same analytic axis-distinct field

    f(x, y, z) = x + 100 y + 10000 z

via `WRITES interior` + `SYNC` at `initial` and `postregrid`. Because `f` is
linear and `CCTK_INITIAL` re-runs on every level, each level's interior holds
`f` exactly (independent of prolongation); interior+SYNC (not "everywhere")
also keeps validity tracking consistent so the poison check after
`MakeNewLevelFromCoarse` doesn't fire on not-yet-synced ghosts. The distinct
decade weights make any axis swap/transpose visible.

| Parfile (`TestPlanesX/test/`) | Coverage |
|---|---|
| `planes-single-level.par` | uniform grid; 3 principal planes + 1 offset/axis; all 8 centerings; both writers |
| `planes-amr.par` | 3-level AMR (refined cube); every plane crosses every level (per-level snap, mrgtree shadowing, openPMD AMR attrs) |
| `planes-amr-midlevel.par` | nested 3-level AMR; planes crossing only a subset of levels |
| `planes-edge-cases.par` | half-integer / negative / over-precise / out-of-domain elevations |
| `planes-int-tags.par` | `planes_frac_precision = 0` integer tags; malformed specs |

The grid is small with a **distinct cell count and non-zero origin per axis**
(`16×20×24`, `x∈[-4,12]`, `y∈[-2,18]`, `z∈[-6,18]`, `dx=1`): distinct sizes
turn an in-plane transpose into a shape/value mismatch and the offsets catch
`ProbLo==0` bugs. `max_grid_size=8`/`blocking_factor=2` force several boxes,
and the single-level + AMR cases run on **2 ranks**, exercising the Silo MPI
gather / multi-file metafile and openPMD collective writes. The AMR cube uses
`ddf` prolongation with the poison guard left on.

`verify_planes.py` checks, for every emitted value, that it equals `f` at the
point's reconstructed coordinate — in-plane coords from the file's own mesh
metadata, the normal coord from an independent replay of `snap_to_grid_index`
(so neither can mask the other) — plus **coarse-level coverage** (the interior
points must tile the whole domain plane → catches silently dropped data) and
that out-of-domain planes produced **no** file. Reading is isolated in the
reusable, geometry-free `TestPlanesX/test/plane_readers/` package (one kernel
per format, returning a uniform `Slab` list); `verify_planes.py` owns only the
parfile/geometry and comparison logic.

Both writers are numerically verified. openPMD via `openpmd_api` (pip); Silo
`.silo` (`DB_HDF5`) via the LLNL `Silo` module (apt `python3-silo`, not on
PyPI): its `DBfile` has only `GetToc`/`GetVar`/`GetVarInfo`, so `GetVarInfo`
gives each object's HDF5 component paths (`value0`, `coord0`/`coord1`),
centering and interior range, and `h5py` reads the arrays (reshape `(nb,na)` —
the buffer is axis_a-fastest, so trust the stride, not h5py's 2D shape). Only
interior cells are checked (AMR coarse-fine ghosts are prolongation-filled, not
exact `f`). Without the Silo module it degrades to a smoke check;
`--silo-mode inspect` dumps the layout/API.

A **golden-reference regression** check sits on top: committed binary reference
output under `TestPlanesX/test/golden/<parfile>/`, generated by
`scripts/make-golden-planes.sh` (runs only the executable — no Python — so it
works on Frontier). Present golden is read back through the same reader and
compared data-level (point→value, keyed on centering, so it is independent of
MPI decomposition and of Silo's per-box variable names); absent golden is a
soft skip. Only the small openPMD `.bp5` golden is committed; the (large)
Silo golden is a stub — when committed it compares the same way.

CI runs `scripts/test-planes.sh` after the build (`.github/workflows/ci.yml`;
also via `agent_scripts/test.sh` locally): it pip-installs `openpmd-api`/`h5py`,
apt-installs `python3-silo`, runs the parfiles, and verifies both writers. To
regenerate golden after an intentional change, run `make-golden-planes.sh` and
commit `TestPlanesX/test/golden/`.

## Why Cartesian-only

`CoordinatesX::vertex_coords` always stores world-space Cartesian (x,y,z), even
on spherical patches — so `z=12` is literal world-z. A constant-z surface
through a curvilinear patch is curved in index space, deferred to a future
interpolation path.

## Relation to CarpetX's own 2D output

Upstream CarpetX (lwJi/dev) later grew an independent 2D slice capability
(`CarpetX::out_silo_2d_*` / `out_openpmd_2d_*` with `IO::out_xyplane_z` etc.,
`io_slice.hxx`). The two coexist: parameter names are disjoint, and PlanesX
keeps the per-level snapping, arbitrary per-plane elevations, mrgtree
shadowing and the TestPlanesX verification described here.

## CarpetX openPMD + HDF5 checkpointing: known issues (review 2026-06-05)

Findings from a review of the checkpoint/recovery path (`CarpetX/src/io.cxx`
`Checkpoint`/`RecoverGH`, `CarpetX/src/io_openpmd.cxx`), done after the
collective-metadata fix (#90, `e538ceec` + `e4e197ff`) landed. Recorded here
because the plane writer mirrors the same openPMD conventions and the HDF5
deadlock class was first found via PlanesX. Metadata rank-symmetry now checks
out: band MultiFab allocation is driven by global BoxArrays
(`driver.cxx:1076-1092`), `write_bands` comes from replicated level
iterations, and the walltime checkpoint trigger is broadcast (`io.cxx:800`).
The following issues remain, none yet fixed:

1. **HDF5 64 KB attribute limit vs. `AllParameters` and `chunkInfo`
   (likely hard failure at scale).** The checkpoint stores the full parameter
   dump as a single string attribute (`io_openpmd.cxx:1564`) and the grid
   structure as one `chunkInfo<patch><level>` int64-vector attribute per level
   (`io_openpmd.cxx:1605`). HDF5 attributes live in the object header
   (≤64 KB per message unless dense attribute storage is enabled, which
   openPMD-api does not do). `chunkInfo` is 48 bytes/box → fails beyond
   ~1,350 boxes on a level; a production ET `AllParameters` dump can also
   exceed 64 KB. ADIOS2 has no such limit, which is why the default
   `openpmd_format = "ADIOS2_auto"` never hits it. (Same hazard class applies
   to the PlanesX per-level `chunkInfo` attributes when writing `.h5` planes.)
   First thing to test: small `max_grid_size`, >1,400 boxes, `.h5` checkpoint.

2. **No checkpoint atomicity, no fallback past a corrupt file.**
   `InputOpenPMDParameters` catches only `openPMD::no_such_file_error`
   (`io_openpmd.cxx:524`) and always recovers the *largest* iteration —
   exactly the file a mid-checkpoint kill truncates. Worse, fileBased series
   parsing opens every `%08T` match, so one corrupt file in `checkpoint_dir`
   aborts recovery even when older good checkpoints exist. No
   write-to-temp-then-rename. HDF5 is the most exposed backend (file is
   unreadable until the metadata flush at close).

3. **Hard dependency on HDF5 *independent* raw-data writes.** Post-#90,
   correctness relies on the openPMD default `OPENPMD_HDF5_INDEPENDENT=ON`:
   the number of H5Dwrite calls per dataset is rank-asymmetric (per-rank fab
   counts, the rank-0-only grid-array data write at `io_openpmd.cxx:2139`,
   sparse subcycling-band chunks). Flipping to collective transfers — a
   common Lustre tuning — deadlocks checkpointing the same way the metadata
   path used to. Should be documented in `param.ccl` or warned about at
   startup.

4. **Singleton state leak on the no-groups early return.** `InputOpenPMD`
   returns early when no group is enabled (`io_openpmd.cxx:723`), *after*
   `InputOpenPMDParameters` left the READ_ONLY series and `read_iter` open in
   `carpetx_openpmd_t::self`; the next `OutputOpenPMD` then sees `series` set
   and dies at `assert(write_iters)` (`io_openpmd.cxx:1543`). The early
   return should close/reset the read state.

5. **Latent ghost-offset bug in the non-GF read expansion.** The in-place
   grid-array expansion (`io_openpmd.cxx:1328-1339`) omits the interior
   offset `box.lo - extbox.lo` from the destination index; the GF path
   applies it (`amrex_offset`, `io_openpmd.cxx:1024-1031`). Unreachable
   today only because grid arrays have zero ghosts (`intbox == extbox`), but
   checkpoint recovery is the code that would hit it.

6. **Scaling caveats.** On recovery every rank parses the metadata of every
   checkpoint file and independently reads the full replicated grid-array
   data (`io_openpmd.cxx:1291-1310`) — N identical reads at N ranks. On
   write: one dataset per (group, patch, level, tl, band) with one
   independent H5Dwrite per fab per variable, unaggregated; the compression
   `options` JSON configures ADIOS2 only (`io_openpmd.cxx:127-155`), so HDF5
   checkpoints are uncompressed. Datasets are sized to the level bounding box
   with fabs writing a sparse subset — fine for chunked HDF5 and for recovery
   (which reads via `chunkInfo`), but generic openPMD readers see fill values
   in uncovered regions of refined levels (same full-extent-chunk issue as
   the plane reader's HDF5 caveat above).

Items 4 and the error handling of 2 are self-contained fixes; 1 needs a
format decision (store as dataset, split, or enable dense attribute
storage upstream in openPMD-api).

## PLANNED: Silo nameschemes (+ CI VisIt gate)

Status: planned, not started. Plan agreed 2026-06-05; execute in this order.

**Motivation.** The Silo metafile stores one explicit string per block in the
multimesh and in *every* multivar — O(blocks × (1 + nvars)) strings of
~150 bytes. At production scale (10³–10⁴ intersecting boxes, tens of
variables, per plane, per iteration) that is megabytes of repeated strings
per metafile, all parsed by VisIt on open. Silo *nameschemes* (≥4.8,
supported by any VisIt of the last decade, incl. 3.3.3/3.4.1) replace the
lists with one printf-like pattern plus small integer index arrays —
metafile size becomes independent of block count. Leaf files and data bytes
are untouched; the numeric verifier (leaf-only) and the value-keyed golden
are unaffected by construction.

**History/why gated.** Three metafile-class bugs (the mrgtree empty-child-map
VisIt crash, the 3D mrgTree name mismatch, the interior-mode
multimesh/multivar block-count mismatch) were invisible to the numeric CI
because nothing opens the metafile; each was caught only by driving VisIt.
A namescheme bug would fail the same way. Hence: the CI VisIt gate lands
FIRST, the feature second.

### Commit A — CI VisIt gate: DONE

The cpu/real64/optimize job installs a cached headless VisIt 3.4.2 (debian12
tarball; the carpetx container additionally needs the system
X11/fontconfig/GL/dbus layer via apt, an `ldd` diagnostic scan, and a
fast-failing cli+viewer smoke test) and runs
`visit -nowin -cli -s scripts/visit-check-planes.py $PLANES_DIR`, time-bounded
with `timeout` (a wedged viewer must not pin the job). VisIt's CLI mangles
exit codes (250 observed on success), so the step gates on the script's
printed "checked N databases, 0 failure(s)" line. Rendered PNGs are uploaded
as the `plane-renders` artifact. The checker opens every metafile index,
every leaf file, and every openPMD `.h5`, and fails draws whose actual data
has zero zones or a non-finite MinMax (the all-white mode). Proven in anger:
it caught the interior-mode empty-leaf bug (see FIXED above) on its first
in-CI run.

### Commit B — `PlanesX::silo_planes_nameschemes`: DONE (default `yes` after the production soak)

BOOLEAN parameter; metafile pass of `silo_planes.cxx` only. Per multimesh,
the per-block integer arrays `<multimesh>_ns_{patch,level,comp,file}`
(`file[n] = s.proc / ioproc_every`; the mapping is irregular — only
intersecting components are listed — so external arrays are required) are
`DBWrite`n into the metafile, and `DBPutMultimesh`/`DBPutMultivar` get null
name arrays plus `DBOPT_MB_FILE_NS` / `DBOPT_MB_BLOCK_NS` patterns such as

    |<subdir>/<sim>.<tag>.it<NNNNNNNN>.p%06d.silo|#/<multimesh>_ns_file[n]
    |box_<tag>[_<ctag>]_ghostsGG_GG_m%04d_rl%02d_c%08d|#/<mm>_ns_patch[n]|...

**Verify-or-fallback.** Rather than deriving the patterns
correct-by-construction (legalization, '%'/'|' in names, format drift), the
writer expands every namescheme through the same file-based
`DBMakeNamescheme(str, 0, dbfile, 0)` path VisIt uses on read and compares
all expansions against the explicit name lists (which are still built — the
win is metafile size and VisIt parse time, not writer CPU; the check is one
sprintf+compare per block, the same order as building the names). On any
mismatch or constructor failure the explicit lists are written instead, with
a warning — a namescheme only ever lands in a file after being proven
equivalent. A multivar reuses the multimesh's arrays only when its block
count matches (out-of-range external-array indexing is undefined); content
agreement is the expansion check's job. mrgtree, `DBOPT_EXTENTS`/
`DBOPT_ZONECOUNTS`, region maps (block indices, not names) and leaf files
are unchanged; the numeric verifier reads leaf files only.

Tests: `planes-options.par` covers the single-level plain-multimesh form and
`planes-amr-noghosts.par` the namescheme × AMR mrgtree × interior-mode
combination; `planes-amr.par` pins the legacy ghost-extended +
explicit-name-list form so both metafile forms stay covered by the CI VisIt
gate. After a Frontier production soak (ghosts=yes and =no output with
namescheme metafiles, visualised in VisIt client/server, 2026-06-06) the
defaults were flipped: `silo_planes_ghosts = no` and
`silo_planes_nameschemes = yes`.

### Commit C (later, optional) — upstream port

The same string-list explosion is larger in upstream's 3D `io_silo.cxx`
metafiles (more boxes, more variables; same structure). After the PlanesX
implementation has soaked, port it as an upstream PR — same proving-ground
pattern as the IO-method hook (lwJi/CarpetX#87) and the openPMD-HDF5
collective fix (#90).
