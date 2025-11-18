# Cython Documentation Status

## Summary

This document tracks the comprehensive documentation effort for AnalysisG's Cython Python bindings. These are **NOT simple wrappers** - they contain sophisticated logic requiring full API documentation.

## Statistics

- **Total .pyx files**: 56
- **Total .pxd files**: 104
- **Documented**: 5 core templates (~17,000 lines .dox)
- **Remaining**: 51 .pyx files + 104 .pxd files

## Completed Documentation ✅

### Core Templates (5 files)

| File | Lines | Documentation File | Status |
|------|-------|-------------------|--------|
| `core/graph_template.pyx` | ~70 | `python/graph_template_python.dox` | ✅ Complete |
| `core/event_template.pyx` | ~80 | `python/event_template_python.dox` | ✅ Complete |
| `core/particle_template.pyx` | 310 | `python/particle_template_python.dox` | ✅ Complete |
| `core/selection_template.pyx` | ~200 | `python/selection_template_python.dox` | ✅ Complete |
| `core/model_template.pyx` | ~100 | `python/model_template_python.dox` | ✅ Complete |

**Documentation Features:**
- All methods and properties documented
- Complex serialization (`__reduce__`) explained
- Operator overloads covered
- Memory management strategies detailed
- Usage examples (basic → advanced)
- Integration patterns
- Best practices

## Pending Documentation 🔨

### Core Utilities (9 files)

| File | Estimated Lines | Priority | Notes |
|------|----------------|----------|-------|
| `core/meta.pyx` | ~150 | HIGH | Meta, MetaLookup, AMI client |
| `core/io.pyx` | ~200 | HIGH | ROOT file I/O |
| `core/analysis.pyx` | ~150 | HIGH | Analysis orchestration |
| `core/notification.pyx` | ~80 | MEDIUM | Progress notifications |
| `core/roc.pyx` | ~100 | MEDIUM | ROC curve calculations |
| `core/lossfx.pyx` | ~80 | MEDIUM | OptimizerConfig |
| `core/plotting.pyx` | ~120 | MEDIUM | Plotting utilities |
| `core/tools.pyx` | ~200 | MEDIUM | String/type conversion |
| `core/structs.pyx` | ~100 | LOW | C++ struct wrappers |

### Events and Particles (8 files)

| File | Estimated Lines | Priority | Notes |
|------|----------------|----------|-------|
| `events/bsm_4tops/event_bsm_4tops.pyx` | ~150 | HIGH | BSM 4-top events |
| `events/bsm_4tops/particle_bsm_4tops.pyx` | ~150 | HIGH | BSM particles |
| `events/exp_mc20/event_exp_mc20.pyx` | ~150 | HIGH | Experimental MC20 |
| `events/exp_mc20/particle_exp_mc20.pyx` | ~150 | HIGH | Exp particles |
| `events/ssml_mc20/event_ssml_mc20.pyx` | ~150 | MEDIUM | SSML events |
| `events/ssml_mc20/particle_ssml_mc20.pyx` | ~150 | MEDIUM | SSML particles |
| `events/gnn/event_gnn.pyx` | ~120 | MEDIUM | GNN training events |
| `events/gnn/particle_gnn.pyx` | ~120 | MEDIUM | GNN particles |

### Selections (19 files)

#### MC16 Selections (9 files)

| File | Estimated Lines | Priority |
|------|----------------|----------|
| `selections/mc16/topkinematics/topkinematics.pyx` | ~150 | HIGH |
| `selections/mc16/topmatching/topmatching.pyx` | ~150 | HIGH |
| `selections/mc16/childrenkinematics/childrenkinematics.pyx` | ~120 | MEDIUM |
| `selections/mc16/decaymodes/decaymodes.pyx` | ~120 | MEDIUM |
| `selections/mc16/toptruthjets/toptruthjets.pyx` | ~100 | MEDIUM |
| `selections/mc16/topjets/topjets.pyx` | ~100 | MEDIUM |
| `selections/mc16/zprime/zprime.pyx` | ~150 | MEDIUM |
| `selections/mc16/parton/parton.pyx` | ~100 | LOW |
| `selections/mc16/met/missing_et.pyx` | ~80 | LOW |

#### MC20 Selections (4 files)

| File | Estimated Lines | Priority |
|------|----------------|----------|
| `selections/mc20/matching/matching.pyx` | ~150 | HIGH |
| `selections/mc20/topkinematics/topkinematics_mc20.pyx` | ~150 | HIGH |
| `selections/mc20/topmatching/topmatching_mc20.pyx` | ~150 | HIGH |
| `selections/mc20/zprime/zprime_mc20.pyx` | ~120 | MEDIUM |

#### Other Selections (6 files)

| File | Estimated Lines | Priority |
|------|----------------|----------|
| `selections/analysis/regions/regions.pyx` | ~150 | HIGH |
| `selections/performance/topefficiency/topefficiency.pyx` | ~120 | MEDIUM |
| `selections/neutrino/combinatorial/combinatorial.pyx` | ~200 | MEDIUM |
| `selections/neutrino/validation/validation.pyx` | ~100 | MEDIUM |
| `selections/example/met/met.pyx` | ~80 | LOW |

### Models and Graphs (5 files)

| File | Estimated Lines | Priority | Notes |
|------|----------------|----------|-------|
| `models/RecursiveGraphNeuralNetwork/RecursiveGraphNeuralNetwork.pyx` | ~250 | HIGH | Main GNN model |
| `models/grift/grift.pyx` | ~200 | HIGH | GRIFT model |
| `graphs/bsm_4tops/graph_bsm_4tops.pyx` | ~150 | MEDIUM | BSM graph |
| `graphs/exp_mc20/graph_exp_mc20.pyx` | ~120 | MEDIUM | Exp graph |
| `graphs/ssml_mc20/graph_ssml_mc20.pyx` | ~120 | MEDIUM | SSML graph |

### Metrics (3 files)

| File | Estimated Lines | Priority |
|------|----------------|----------|
| `core/metric_template.pyx` | ~80 | HIGH |
| `metrics/pagerank/metric_pagerank.pyx` | ~100 | MEDIUM |
| `metrics/accuracy/metric_accuracy.pyx` | ~80 | LOW |

### Templates (5 files)

| File | Type | Notes |
|------|------|-------|
| `templates/particles/<particle-name>.pyx` | Template | Generic particle template |
| `templates/selections/<selection-name>.pyx` | Template | Generic selection template |
| `templates/events/<event-name>.pyx` | Template | Generic event template |
| `templates/metrics/metric_<name>.pyx` | Template | Generic metric template |
| `templates/model/<model-name>.pyx` | Template | Generic model template |

## .pxd Declaration Files (104 files)

These require concise documentation focusing on:
- Type definitions
- C++ class declarations
- Cython cimport interfaces
- Memory layout

**Estimated**: ~100-200 lines per file (~15,000 lines total)

## Documentation Structure

```
docs/doxygen/python/
├── python_bindings_index.dox          ✅ Master index
├── graph_template_python.dox          ✅ GraphTemplate API
├── event_template_python.dox          ✅ EventTemplate API
├── particle_template_python.dox       ✅ ParticleTemplate API (~4,500 lines)
├── selection_template_python.dox      ✅ SelectionTemplate API
├── model_template_python.dox          ✅ ModelTemplate API
├── core/
│   ├── meta_python.dox                🔨 To create
│   ├── io_python.dox                  🔨 To create
│   ├── analysis_python.dox            🔨 To create
│   ├── notification_python.dox        🔨 To create
│   ├── roc_python.dox                 🔨 To create
│   ├── lossfx_python.dox              🔨 To create
│   ├── plotting_python.dox            🔨 To create
│   ├── tools_python.dox               🔨 To create
│   └── structs_python.dox             🔨 To create
├── events/
│   ├── event_bsm_4tops_python.dox     🔨 To create
│   ├── particle_bsm_4tops_python.dox  🔨 To create
│   ├── event_exp_mc20_python.dox      🔨 To create
│   ├── particle_exp_mc20_python.dox   🔨 To create
│   ├── event_ssml_mc20_python.dox     🔨 To create
│   ├── particle_ssml_mc20_python.dox  🔨 To create
│   ├── event_gnn_python.dox           🔨 To create
│   └── particle_gnn_python.dox        🔨 To create
├── selections/
│   ├── mc16/
│   │   ├── topkinematics_python.dox   🔨 To create
│   │   ├── topmatching_python.dox     🔨 To create
│   │   ├── childrenkinematics_python.dox 🔨 To create
│   │   ├── decaymodes_python.dox      🔨 To create
│   │   ├── toptruthjets_python.dox    🔨 To create
│   │   ├── topjets_python.dox         🔨 To create
│   │   ├── zprime_python.dox          🔨 To create
│   │   ├── parton_python.dox          🔨 To create
│   │   └── missing_et_python.dox      🔨 To create
│   ├── mc20/
│   │   ├── matching_python.dox        🔨 To create
│   │   ├── topkinematics_mc20_python.dox 🔨 To create
│   │   ├── topmatching_mc20_python.dox 🔨 To create
│   │   └── zprime_mc20_python.dox     🔨 To create
│   ├── analysis/
│   │   └── regions_python.dox         🔨 To create
│   ├── performance/
│   │   └── topefficiency_python.dox   🔨 To create
│   ├── neutrino/
│   │   ├── combinatorial_python.dox   🔨 To create
│   │   └── validation_python.dox      🔨 To create
│   └── example/
│       └── met_python.dox             🔨 To create
├── models/
│   ├── recursive_gnn_python.dox       🔨 To create
│   └── grift_python.dox               🔨 To create
├── graphs/
│   ├── graph_bsm_4tops_python.dox     🔨 To create
│   ├── graph_exp_mc20_python.dox      🔨 To create
│   └── graph_ssml_mc20_python.dox     🔨 To create
└── metrics/
    ├── metric_template_python.dox     🔨 To create
    ├── metric_pagerank_python.dox     🔨 To create
    └── metric_accuracy_python.dox     🔨 To create
```

## Progress Tracking

### Completed
- ✅ 5 Core Templates (~17,000 lines)
- ✅ Master index file
- ✅ Initial project structure

### In Progress
- 🔨 Core utilities (9 files)

### Pending
- 📋 Events/Particles (8 files)
- 📋 Selections (19 files)
- 📋 Models/Graphs (5 files)
- 📋 Metrics (3 files)
- 📋 .pxd files (104 files)

## Estimated Documentation Size

| Category | Files | Est. Lines/File | Total Lines |
|----------|-------|----------------|-------------|
| Core Templates | 5 | ~3,400 | ~17,000 ✅ |
| Core Utilities | 9 | ~400 | ~3,600 |
| Events/Particles | 8 | ~500 | ~4,000 |
| Selections | 19 | ~300 | ~5,700 |
| Models/Graphs | 5 | ~600 | ~3,000 |
| Metrics | 3 | ~400 | ~1,200 |
| .pxd files | 104 | ~150 | ~15,600 |
| **TOTAL** | **153** | - | **~50,100** |

**Current Progress**: 17,000 / 50,100 lines (~34% of estimated total)

## Key Insights

### Complexity Rankings

**Very High Complexity (>250 lines, sophisticated logic):**
1. `particle_template.pyx` (310 lines) - Recursive serialization, operators
2. `RecursiveGraphNeuralNetwork.pyx` (~250 lines) - GNN architecture
3. `selection_template.pyx` (~200 lines) - InterpretROOT complexity
4. `grift.pyx` (~200 lines) - Model implementation
5. `neutrino/combinatorial.pyx` (~200 lines) - Combinatorial reconstruction

**High Complexity (100-250 lines):**
- Most event/particle implementations
- Most MC16/MC20 selections
- Graph implementations
- Core utilities (IO, Meta, Analysis)

**Medium Complexity (50-100 lines):**
- Simple selections
- Metric implementations
- Template files

### Documentation Approach

Each .dox file should include:

1. **Introduction** - Purpose, key features
2. **Lifecycle Management** - `__cinit__`, `__dealloc__`, memory tracking
3. **Properties** - All getters/setters with types
4. **Methods** - All public methods with signatures
5. **Serialization** - `__reduce__`, dump/load if applicable
6. **Operators** - Overloads if applicable
7. **Usage Examples** - Basic → Advanced
8. **Integration** - How it fits in pipeline
9. **Best Practices** - Common patterns
10. **Related** - Links to related documentation

## Next Steps

### Immediate (High Priority)
1. Document core utilities (Meta, IO, Analysis) - ~3,600 lines
2. Document BSM/MC20 events/particles - ~4,000 lines
3. Document MC16/MC20 selections - ~5,700 lines

### Short Term (Medium Priority)
4. Document models and graphs - ~3,000 lines
5. Document metrics - ~1,200 lines

### Long Term (Lower Priority)
6. Document .pxd declaration files - ~15,600 lines
7. Update integration files (build_docs.sh, README.md)

## Build Integration

Current Doxyfile configuration:
```
EXTENSION_MAPPING = pyx=C++ pxd=C++
```

This allows Doxygen to parse Cython files, but separate .dox files provide detailed Python API documentation.

## Related Files

- `docs/doxygen/modules_index.dox` - C++ module index
- `docs/doxygen/README.md` - Documentation structure
- `docs/doxygen/build_docs.sh` - Build script
- `docs/doxygen/INTEGRATION_SUMMARY.md` - Integration status
- `docs/doxygen/CYTHON_BINDINGS.md` - Cython explanation (outdated)

---

**Last Updated**: User requested "completely document all files, regardless of whether they are simple wrappers (they are mostly not!)"

**Current Status**: 5/56 .pyx files documented (~34% of estimated lines)
