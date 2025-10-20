#!/usr/bin/env python3
"""
Generate RST files for all modules in src/AnalysisG to integrate with Doxygen/Breathe
"""

import os
from pathlib import Path
from collections import defaultdict

def get_all_source_files():
    """Get all source files in src/AnalysisG organized by module"""
    src_dir = Path("../src/AnalysisG")
    modules = defaultdict(lambda: {'headers': [], 'sources': [], 'python': []})
    
    # Walk through all directories
    for root, dirs, files in os.walk(src_dir):
        rel_path = Path(root).relative_to(src_dir)
        
        # Skip certain directories
        if any(skip in str(rel_path) for skip in ['__pycache__', '.git', 'build', 'dist']):
            continue
            
        for file in files:
            # Get path relative to AnalysisG directory (what Doxygen sees after STRIP_FROM_PATH)
            rel_to_analysisg = Path(root).relative_to(src_dir) / file
            
            if file.endswith('.h'):
                modules[str(rel_path.parts[0] if len(rel_path.parts) > 0 else 'root')]['headers'].append(
                    str(rel_to_analysisg)
                )
            elif file.endswith('.cxx'):
                modules[str(rel_path.parts[0] if len(rel_path.parts) > 0 else 'root')]['sources'].append(
                    str(rel_to_analysisg)
                )
            elif file.endswith(('.py', '.pyx', '.pxd')):
                modules[str(rel_path.parts[0] if len(rel_path.parts) > 0 else 'root')]['python'].append(
                    str(rel_to_analysisg)
                )
    
    return modules

def generate_module_rst(module_name, files, output_dir):
    """Generate RST file for a module"""
    output_path = output_dir / f"{module_name}.rst"
    
    # Create a nice title
    title = module_name.replace('_', ' ').title()
    title_underline = '=' * len(title)
    
    content = f"""{title}
{title_underline}

This section documents the {module_name} module.

"""
    
    # Add headers section
    if files['headers']:
        content += "Header Files\n"
        content += "-" * 12 + "\n\n"
        for header in sorted(files['headers']):
            # Path is already relative to AnalysisG directory
            content += f".. doxygenfile:: {header}\n"
            content += f"   :project: AnalysisG\n\n"
    
    # Add source files section
    if files['sources']:
        content += "Source Files\n"
        content += "-" * 12 + "\n\n"
        for source in sorted(files['sources']):
            # Path is already relative to AnalysisG directory
            content += f".. doxygenfile:: {source}\n"
            content += f"   :project: AnalysisG\n\n"
    
    # Add Python files section (note: Doxygen can parse Python)
    if files['python']:
        content += "Python Files\n"
        content += "-" * 12 + "\n\n"
        for pyfile in sorted(files['python']):
            # Path is already relative to AnalysisG directory
            content += f".. doxygenfile:: {pyfile}\n"
            content += f"   :project: AnalysisG\n\n"
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    return output_path.name

def generate_api_index(module_names, output_dir):
    """Generate the main API index RST file"""
    output_path = output_dir / "api_reference.rst"
    
    content = """API Reference
=============

Complete API documentation for all AnalysisG modules.

.. toctree::
   :maxdepth: 2
   :caption: Modules

"""
    
    for module_name in sorted(module_names):
        content += f"   api/{module_name}\n"
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    return output_path.name

def main():
    """Main function"""
    # Create API directory if it doesn't exist
    api_dir = Path("source/api")
    api_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all source files organized by module
    modules = get_all_source_files()
    
    print(f"Found {len(modules)} modules")
    
    created_files = []
    
    # Generate RST for each module
    for module_name, files in modules.items():
        if any(files.values()):  # Only create if there are files
            print(f"Generating documentation for {module_name}...")
            rst_file = generate_module_rst(module_name, files, api_dir)
            created_files.append(f"docs/source/api/{rst_file}")
    
    # Generate main API index
    print("Generating API index...")
    index_file = generate_api_index(modules.keys(), Path("source"))
    created_files.append(f"docs/source/{index_file}")
    
    # Update the API index.rst
    api_index_path = api_dir / "index.rst"
    api_index_content = """API Documentation
=================

Complete API reference for all AnalysisG modules.

.. toctree::
   :maxdepth: 2
   :caption: Modules:

"""
    for module_name in sorted(modules.keys()):
        if any(modules[module_name].values()):
            api_index_content += f"   {module_name}\n"
    
    with open(api_index_path, 'w') as f:
        f.write(api_index_content)
    created_files.append(f"docs/source/api/index.rst")
    
    print(f"\nCreated {len(created_files)} RST files:")
    for file in sorted(created_files):
        print(f"  - {file}")
    
    return created_files

if __name__ == "__main__":
    main()
