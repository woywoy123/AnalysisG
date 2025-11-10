#!/usr/bin/env python3
"""
Generate .dox files for all source files in the repository.
This script reads actual source files and creates Doxygen documentation.
"""

import os
import re
from pathlib import Path

def extract_file_info(filepath):
    """Extract documentation info from a source file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    info = {
        'classes': [],
        'functions': [],
        'structs': [],
        'brief': '',
        'includes': []
    }
    
    # Extract includes
    includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)
    info['includes'] = includes[:10]  # Limit to first 10
    
    # Extract classes (C++)
    class_matches = re.findall(r'class\s+(\w+)', content)
    info['classes'] = list(set(class_matches))[:20]  # Limit and dedupe
    
    # Extract structs (C++)
    struct_matches = re.findall(r'struct\s+(\w+)', content)
    info['structs'] = list(set(struct_matches))[:20]
    
    # Extract function signatures (simplified)
    func_matches = re.findall(r'(?:def|cdef|cpdef)\s+(\w+)\s*\(', content)
    if not func_matches:
        func_matches = re.findall(r'\b\w+\s+(\w+)\s*\([^)]*\)\s*[;{]', content)
    info['functions'] = list(set(func_matches))[:30]
    
    return info

def generate_dox_content(filepath, rel_path, info):
    """Generate .dox file content."""
    filename = os.path.basename(filepath)
    module_path = rel_path.replace('/', '::').replace('.pyx', '').replace('.cxx', '').replace('.cu', '').replace('.h', '').replace('.cuh', '')
    
    dox_content = f"""/**
 * @file {filename}
 * @brief Documentation for {rel_path}
 * 
 * This file is part of the AnalysisG framework.
 * 
 * @details
 * Source file: `{rel_path}`
 * 
"""
    
    if info['brief']:
        dox_content += f" * @section overview Overview\n * {info['brief']}\n *\n"
    
    if info['includes']:
        dox_content += " * @section dependencies Dependencies\n"
        for inc in info['includes']:
            dox_content += f" * - {inc}\n"
        dox_content += " *\n"
    
    if info['classes']:
        dox_content += " * @section classes Classes\n"
        for cls in info['classes']:
            dox_content += f" * - {cls}\n"
        dox_content += " *\n"
    
    if info['structs']:
        dox_content += " * @section structs Structures\n"
        for struct in info['structs']:
            dox_content += f" * - {struct}\n"
        dox_content += " *\n"
    
    if info['functions']:
        dox_content += " * @section functions Functions\n"
        for func in info['functions'][:20]:  # Limit display
            dox_content += f" * - {func}()\n"
        dox_content += " *\n"
    
    dox_content += f""" * @see {filename}
 * @ingroup {module_path.split('::')[0] if '::' in module_path else 'core'}
 */
"""
    
    return dox_content

def main():
    """Generate all .dox files."""
    repo_root = Path('/home/runner/work/AnalysisG/AnalysisG')
    src_root = repo_root / 'src' / 'AnalysisG'
    dox_dir = repo_root / 'docs' / 'doxygen_pages'
    
    # Find all source files
    patterns = ['*.pyx', '*.cxx', '*.cu', '*.h', '*.cuh']
    source_files = []
    
    for pattern in patterns:
        source_files.extend(src_root.rglob(pattern))
    
    # Filter out selections and templates
    source_files = [f for f in source_files 
                   if 'selections' not in str(f) and 'templates' not in str(f)]
    
    print(f"Found {len(source_files)} source files to document")
    
    generated_count = 0
    
    for filepath in sorted(source_files):
        rel_path = filepath.relative_to(src_root)
        
        # Create .dox filename
        dox_name = str(rel_path).replace('/', '_').replace('.', '_') + '.dox'
        dox_path = dox_dir / dox_name
        
        # Extract info from source file
        try:
            info = extract_file_info(filepath)
            
            # Generate .dox content
            dox_content = generate_dox_content(filepath, str(rel_path), info)
            
            # Write .dox file
            with open(dox_path, 'w') as f:
                f.write(dox_content)
            
            generated_count += 1
            
            if generated_count % 50 == 0:
                print(f"Generated {generated_count} .dox files...")
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    print(f"\nSuccessfully generated {generated_count} .dox files in {dox_dir}")
    
    # Create index file
    index_content = """/**
 * @mainpage AnalysisG Documentation
 * 
 * @section intro Introduction
 * Complete API documentation for the AnalysisG framework.
 * 
 * @section modules Modules
 * - @ref core - Core Python/Cython modules
 * - @ref events - Event data structures
 * - @ref graphs - Graph representations
 * - @ref metrics - Performance metrics
 * - @ref models - Machine learning models
 * - @ref pyc - C++/CUDA interface modules
 * 
 * @section overview Framework Overview
 * AnalysisG is a high-performance analysis framework combining:
 * - Python/Cython for high-level interfaces
 * - C++ for computational kernels
 * - CUDA for GPU acceleration
 * - ROOT for data I/O
 * - PyTorch for machine learning
 */
"""
    
    with open(dox_dir / 'index.dox', 'w') as f:
        f.write(index_content)
    
    print("Created index.dox")

if __name__ == '__main__':
    main()
