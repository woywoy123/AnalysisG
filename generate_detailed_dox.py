#!/usr/bin/env python3
"""
Generate detailed .dox files alongside source files.
Documents all classes, functions, member variables, and input parameters.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def extract_class_details(content: str, class_name: str) -> Dict:
    """Extract detailed information about a class."""
    details = {
        'methods': [],
        'members': [],
        'base_classes': []
    }
    
    # Find class definition and body
    class_pattern = rf'class\s+{class_name}(?:\s*\(([^)]+)\))?(?:\s*:)?'
    class_match = re.search(class_pattern, content)
    
    if class_match and class_match.group(1):
        details['base_classes'] = [b.strip() for b in class_match.group(1).split(',')]
    
    # Extract methods (Python/Cython style)
    method_pattern = r'(?:def|cdef|cpdef)\s+(\w+)\s*\(([^)]*)\)'
    methods = re.findall(method_pattern, content)
    for method_name, params in methods:
        details['methods'].append({
            'name': method_name,
            'params': [p.strip() for p in params.split(',') if p.strip()]
        })
    
    # Extract member variables (C++ style)
    member_pattern = r'(?:public|private|protected):\s*\n\s*(\w+(?:\s*\*)?)\s+(\w+);'
    members = re.findall(member_pattern, content)
    for member_type, member_name in members:
        details['members'].append({
            'type': member_type.strip(),
            'name': member_name
        })
    
    # Extract Cython attributes
    attr_pattern = r'cdef\s+(?:public\s+)?(\w+)\s+(\w+)'
    attrs = re.findall(attr_pattern, content)
    for attr_type, attr_name in attrs:
        if attr_name not in [m['name'] for m in details['members']]:
            details['members'].append({
                'type': attr_type,
                'name': attr_name
            })
    
    return details

def extract_function_signature(content: str, func_name: str) -> Dict:
    """Extract detailed function signature including parameters."""
    # Python/Cython style
    pattern = rf'(?:def|cdef|cpdef)\s+{func_name}\s*\(([^)]*)\)(?:\s*->\s*(\w+))?'
    match = re.search(pattern, content)
    
    if match:
        params_str = match.group(1)
        return_type = match.group(2) if match.group(2) else 'void'
        
        params = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                if ':' in param:  # Type annotation
                    name, ptype = param.split(':', 1)
                    params.append({'name': name.strip(), 'type': ptype.strip()})
                else:
                    params.append({'name': param, 'type': 'auto'})
        
        return {
            'params': params,
            'return_type': return_type
        }
    
    # C++ style
    cpp_pattern = rf'(\w+)\s+{func_name}\s*\(([^)]*)\)'
    cpp_match = re.search(cpp_pattern, content)
    
    if cpp_match:
        return_type = cpp_match.group(1)
        params_str = cpp_match.group(2)
        
        params = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                parts = param.split()
                if len(parts) >= 2:
                    params.append({'name': parts[-1].strip('*&'), 'type': ' '.join(parts[:-1])})
                elif parts:
                    params.append({'name': parts[0], 'type': 'auto'})
        
        return {
            'params': params,
            'return_type': return_type
        }
    
    return {'params': [], 'return_type': 'void'}

def extract_comprehensive_info(filepath: Path) -> Dict:
    """Extract comprehensive documentation from source file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    info = {
        'classes': {},
        'functions': {},
        'structs': {},
        'includes': [],
        'brief': '',
        'enums': []
    }
    
    # Extract includes
    includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)
    info['includes'] = list(set(includes))
    
    # Extract imports (Python/Cython)
    imports = re.findall(r'(?:from|import)\s+([\w.]+)', content)
    info['includes'].extend(imports[:20])
    
    # Extract classes with details
    class_names = re.findall(r'(?:class|cdef\s+class)\s+(\w+)', content)
    for class_name in set(class_names):
        info['classes'][class_name] = extract_class_details(content, class_name)
    
    # Extract structs
    struct_names = re.findall(r'(?:struct|cdef\s+struct)\s+(\w+)', content)
    for struct_name in set(struct_names):
        info['structs'][struct_name] = extract_class_details(content, struct_name)
    
    # Extract functions
    func_pattern = r'(?:def|cdef|cpdef)\s+(\w+)\s*\('
    func_names = re.findall(func_pattern, content)
    for func_name in set(func_names):
        if func_name not in ['__init__', '__del__', '__repr__', '__str__']:
            info['functions'][func_name] = extract_function_signature(content, func_name)
    
    # C++ functions
    cpp_func_pattern = r'\b(\w+)\s+(\w+)\s*\([^)]*\)\s*[{;]'
    cpp_funcs = re.findall(cpp_func_pattern, content)
    for return_type, func_name in cpp_funcs:
        if func_name not in info['functions'] and return_type in ['void', 'int', 'float', 'double', 'bool', 'std::string', 'torch::Tensor']:
            info['functions'][func_name] = extract_function_signature(content, func_name)
    
    # Extract enums
    enum_pattern = r'enum\s+(?:class\s+)?(\w+)'
    enums = re.findall(enum_pattern, content)
    info['enums'] = list(set(enums))
    
    # Try to extract brief description from docstring or comments
    docstring = re.search(r'"""([^"]+)"""', content)
    if docstring:
        info['brief'] = docstring.group(1).strip().split('\n')[0][:200]
    else:
        comment = re.search(r'/\*\*?\s*\n?\s*\*?\s*([^\n]+)', content)
        if comment:
            info['brief'] = comment.group(1).strip()[:200]
    
    return info

def generate_detailed_dox(filepath: Path, info: Dict) -> str:
    """Generate detailed .dox content."""
    filename = filepath.name
    
    dox = f"""/**
 * @file {filename}
 * @brief {info['brief'] if info['brief'] else f'Implementation for {filename}'}
 * 
 * @details
 * This file contains the implementation of various components for the AnalysisG framework.
 * 
 * **Location**: `{filepath.relative_to(Path('/home/runner/work/AnalysisG/AnalysisG/src/AnalysisG'))}`
 *
"""
    
    # Dependencies
    if info['includes']:
        dox += " * @section dependencies Dependencies\n *\n"
        for inc in sorted(set(info['includes']))[:30]:
            dox += f" * - `{inc}`\n"
        dox += " *\n"
    
    # Classes
    if info['classes']:
        dox += " * @section classes Classes\n *\n"
        for class_name, details in sorted(info['classes'].items()):
            dox += f" * @subsection class_{class_name} {class_name}\n"
            
            if details['base_classes']:
                dox += f" * - **Inherits from**: {', '.join(details['base_classes'])}\n"
            
            if details['members']:
                dox += " * - **Member Variables**:\n"
                for member in details['members'][:20]:
                    dox += f" *   - `{member['type']} {member['name']}`\n"
            
            if details['methods']:
                dox += " * - **Methods**:\n"
                for method in details['methods'][:30]:
                    params_str = ', '.join([p if isinstance(p, str) else p for p in method['params'][:10]])
                    dox += f" *   - `{method['name']}({params_str if len(params_str) < 80 else '...'})`\n"
            
            dox += " *\n"
    
    # Structs
    if info['structs']:
        dox += " * @section structs Structures\n *\n"
        for struct_name, details in sorted(info['structs'].items()):
            dox += f" * @subsection struct_{struct_name} {struct_name}\n"
            
            if details['members']:
                dox += " * - **Members**:\n"
                for member in details['members'][:20]:
                    dox += f" *   - `{member['type']} {member['name']}`\n"
            
            dox += " *\n"
    
    # Functions
    if info['functions']:
        dox += " * @section functions Functions\n *\n"
        for func_name, signature in sorted(info['functions'].items())[:40]:
            params_list = ', '.join([f"{p['type']} {p['name']}" for p in signature['params'][:10]])
            if len(params_list) > 100:
                params_list = params_list[:100] + "..."
            
            dox += f" * @subsection func_{func_name} {func_name}\n"
            dox += f" * - **Signature**: `{signature['return_type']} {func_name}({params_list})`\n"
            
            if signature['params']:
                dox += " * - **Parameters**:\n"
                for param in signature['params'][:15]:
                    dox += f" *   - `{param['name']}` ({param['type']})\n"
            
            dox += f" * - **Returns**: `{signature['return_type']}`\n"
            dox += " *\n"
    
    # Enums
    if info['enums']:
        dox += " * @section enums Enumerations\n *\n"
        for enum in sorted(info['enums']):
            dox += f" * - `{enum}`\n"
        dox += " *\n"
    
    dox += f""" * @see {filename}
 */
"""
    
    return dox

def main():
    """Generate .dox files alongside source files."""
    repo_root = Path('/home/runner/work/AnalysisG/AnalysisG')
    src_root = repo_root / 'src' / 'AnalysisG'
    
    # Find all source files
    patterns = ['*.pyx', '*.cxx', '*.cu', '*.h', '*.cuh', '*.cpp']
    source_files = []
    
    for pattern in patterns:
        source_files.extend(src_root.rglob(pattern))
    
    # Filter out selections and templates
    source_files = [f for f in source_files 
                   if 'selections' not in str(f) and 'templates' not in str(f)]
    
    print(f"Found {len(source_files)} source files to document")
    
    generated_count = 0
    errors = []
    
    for filepath in sorted(source_files):
        try:
            # Extract comprehensive info
            info = extract_comprehensive_info(filepath)
            
            # Generate detailed .dox content
            dox_content = generate_detailed_dox(filepath, info)
            
            # Create .dox file alongside source file
            dox_path = filepath.with_suffix(filepath.suffix + '.dox')
            
            with open(dox_path, 'w') as f:
                f.write(dox_content)
            
            generated_count += 1
            
            if generated_count % 50 == 0:
                print(f"Generated {generated_count} .dox files...")
                
        except Exception as e:
            errors.append(f"{filepath}: {e}")
            continue
    
    print(f"\n✅ Successfully generated {generated_count} .dox files")
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors encountered:")
        for err in errors[:10]:
            print(f"  - {err}")
    
    print(f"\n📁 .dox files placed alongside source files in src/AnalysisG/")

if __name__ == '__main__':
    main()
