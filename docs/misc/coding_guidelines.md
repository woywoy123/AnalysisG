# Coding Guidelines

# AnalysisG Coding Guidelines

Follow these guidelines to ensure consistent code quality in the AnalysisG framework.

## General Principles

-   **Readability First**: Write clear code, even if it's less clever.
-   **Documentation**: Document complex parts and important decisions.
-   **Test Coverage**: Write tests for all new features.
-   **Compatibility**: Ensure code works on supported platforms.

## C++ Guidelines

### Naming

-   **Classes/Structs**: `PascalCase` (e.g., `EventTemplate`)
-   **Functions/Methods**: `snake_case` (e.g., `process_event`)
-   **Variables**: `snake_case` (e.g., `event_count`)
-   **Constants/Enums**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_EVENTS`)
-   **Template Parameters**: `PascalCase` with 'T' prefix (e.g., `TEventType`)
-   **Private Members**: `snake_case` with `_` suffix (e.g., `event_count_`)

### File Organization

-   **Header/Source**: One class per `.h` and `.cpp` file.
-   **Naming**: Filenames match class names (e.g., `EventTemplate.h`).
-   **Include Guards**: Use `#pragma once`.

```cpp
// EventTemplate.h
#pragma once

namespace AnalysisG {
    class EventTemplate {
        // ...
    };
}
```

### Code Style

#### Formatting

-   Use 4 spaces for indentation (no tabs).
-   Opening braces `{` on the same line.
-   Closing braces `}` on a new line.
-   Max line length: 100 characters.

```cpp
if (condition) {
    do_something();
} else {
    do_something_else();
}
```

#### Comments

-   Use Doxygen style for public APIs.
-   Use inline comments sparingly for non-obvious code.

```cpp
/**
 * @brief Processes an event and extracts features.
 *
 * @param event The event to process.
 * @return true if processing succeeded, false otherwise.
 */
bool process_event(const Event& event);
```

#### Best Practices

-   Use `nullptr` instead of `NULL` or `0`.
-   Use `const` whenever possible.
-   Prefer `enum class` over plain `enum`.
-   Use `override` and `final` for virtual functions.
-   Prefer smart pointers and references over raw pointers.

```cpp
// Good
std::shared_ptr<Graph> create_graph() const;

// Avoid
Graph* create_graph() const;
```

### Error Handling

-   Use exceptions for unexpected errors.
-   Document exceptions thrown by public functions.
-   Create custom exceptions inheriting from `std::exception`.

```cpp
class GraphException : public std::exception {
public:
    explicit GraphException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};
```

## Python Guidelines

### Naming

-   **Modules/Packages**: `lowercase_with_underscores`
-   **Classes**: `PascalCase`
-   **Functions/Methods**: `snake_case`
-   **Variables**: `snake_case`
-   **Constants**: `SCREAMING_SNAKE_CASE`

### Code Style

-   Follow PEP 8.
-   Max line length: 88 characters (black compatible).
-   Use NumPy/Google style docstrings.

```python
def process_graph(graph, features=None):
    """Processes a graph and extracts features.

    Args:
        graph (nx.Graph): The graph to process.
        features (list, optional): List of features to extract.

    Returns:
        dict: A dictionary with extracted features.

    Raises:
        ValueError: If the graph is empty.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph cannot be empty")

    # Implementation...
```

### Best Practices

-   Use type hints.
-   Prefer list comprehensions over `map` and `filter`.
-   Use `pathlib` for path manipulation.
-   Use `with` statements for resource management.

```python
from pathlib import Path
from typing import List, Dict, Optional
import networkx as nx # Assuming nx is used for graphs

def read_graphs(directory: Path, pattern: str = "*.graphml") -> List[nx.Graph]:
    """Reads all graphs from a directory.

    Args:
        directory: Directory containing graph files.
        pattern: Glob pattern for files to read.

    Returns:
        List of loaded graphs.
    """
    graphs = []
    for file_path in directory.glob(pattern):
        with open(file_path, 'r') as f:
            # Implementation to load graph...
            pass
    return graphs
```

## Unit Tests

### C++ Tests

-   Use Google Test or Catch2.
-   Test all public functions/methods.
-   Organize tests logically.
-   Use fixtures for setup.

```cpp
#include "gtest/gtest.h" // Example include
#include "AnalysisG/EventTemplate.h" // Include necessary headers
#include "AnalysisG/GraphBuilder.h"
#include "AnalysisG/Graph.h"

TEST(GraphBuilderTest, EmptyEventReturnsEmptyGraph) {
    AnalysisG::EventTemplate empty_event;
    AnalysisG::GraphBuilder builder;

    AnalysisG::Graph graph = builder.build(empty_event);

    EXPECT_EQ(0, graph.get_num_nodes());
    EXPECT_EQ(0, graph.get_num_edges());
}
```

### Python Tests

-   Use pytest.
-   Organize tests by functionality.
-   Use fixtures for shared resources.
-   Use parameterized tests.

```python
import pytest
import numpy as np
# Assume GraphClassifier exists in AnalysisG.ml
# from AnalysisG.ml import GraphClassifier

# Placeholder class until GraphClassifier is defined
class GraphClassifier:
    def __init__(self, hidden_dim, num_layers):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    def __call__(self, x, edge_index):
        # Dummy implementation
        return np.random.randn(1, 2)

class TestGraphClassifier:
    @pytest.fixture
    def model(self):
        return GraphClassifier(hidden_dim=64, num_layers=3)

    def test_forward_pass(self, model):
        x = np.random.randn(10, 5)  # 10 nodes, 5 features
        edge_index = np.array([[0, 1, 2], [1, 2, 3]]) # 3 edges

        output = model(x, edge_index)

        assert output.shape == (1, 2) # Binary classification output shape
```

## Performance Considerations

-   Use `const` references for large function parameters.
-   Avoid unnecessary object copies.
-   Use move semantics (C++11+).
-   Use static array sizes when known.
-   Profile code to find bottlenecks.

```cpp
// Good
void process_graph(const Graph& graph);

// Avoid
void process_graph(Graph graph);
```

## Continuous Integration (CI)

-   All pull requests must pass CI checks.
-   CI includes:
    -   Builds on all supported platforms.
    -   Unit and integration tests.
    -   Linting (clang-format for C++, flake8/black for Python).
    -   Documentation generation.

## Code Review Process

1.  Fork and clone the repository.
2.  Create a feature branch.
3.  Implement and test changes.
4.  Submit a pull request.
5.  Undergo code review (at least one approval needed).
6.  Address feedback.
7.  A team member merges the pull request.

## Versioning

-   Follow [Semantic Versioning](https://semver.org/):
    -   MAJOR: Breaking API changes.
    -   MINOR: Backward-compatible new features.
    -   PATCH: Backward-compatible bug fixes.

Following these guidelines helps keep AnalysisG maintainable and understandable. Thanks for contributing!