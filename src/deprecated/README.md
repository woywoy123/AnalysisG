# Deprecated Documentation

This folder contains documentation files that were previously scattered throughout the src directory.

These files have been moved here as part of the documentation restructuring to use Doxygen (.dox files) for code documentation.

## Files in this folder:

- `particles_bsm_4tops.rst` - Original RST documentation from events/bsm_4tops
- `eventtemplate.md` - Template documentation for events
- `modeltemplate.md` - Template documentation for models
- `selectiontemplate.md` - Template documentation for selections
- `graphtemplate.md` - Template documentation for graphs
- `particletemplate.md` - Template documentation for particles

## Migration Details

These files were moved on 2024-11-15 as part of the comprehensive documentation restructuring:

### Previous Locations:
- `particles_bsm_4tops.rst` → `src/AnalysisG/events/bsm_4tops/include/bsm_4tops/`
- `eventtemplate.md` → `src/AnalysisG/templates/events/`
- `modeltemplate.md` → `src/AnalysisG/templates/model/`
- `selectiontemplate.md` → `src/AnalysisG/templates/selections/`
- `graphtemplate.md` → `src/AnalysisG/templates/graphs/`
- `particletemplate.md` → `src/AnalysisG/templates/particles/`

### New Documentation System:

The content from these files has been incorporated into:

1. **Doxygen .dox files** in `docs/doxygen/`
   - `templates.dox` - References these template guides
   - `events.dox` - Covers event implementation patterns
   - Other module-specific .dox files

2. **Sphinx RST files** in `docs/source/`
   - `api/templates.rst` - Template API documentation
   - `modules/templates.rst` - Template usage guides
   - `examples.rst` - Usage examples

3. **Source code comments**
   - Doxygen-style comments in header files
   - Inline documentation for classes and functions

## Accessing New Documentation

To view the current documentation:

1. **Build Doxygen docs**: `doxygen Doxyfile` (from repository root)
   - View at: `doxygen-docs/html/index.html`

2. **Build Sphinx docs**: `cd docs && make html`
   - View at: `docs/build/html/index.html`

3. **Online**: Visit the Read the Docs site (when deployed)

## Preserving History

These files are kept for reference and historical purposes. The information they contain has been:
- Reorganized for better discoverability
- Updated for current framework structure
- Integrated with API documentation
- Enhanced with additional examples

If you need information from these deprecated files, check the new documentation first, as it may be more up-to-date.
