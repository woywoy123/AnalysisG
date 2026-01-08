#!/usr/bin/env python3
"""
Script zur Überprüfung und Korrektur der Breathe-Konfiguration in der Sphinx-Dokumentation.
"""

import os
import sys
from pathlib import Path
import re
import shutil

def find_conf_py(base_dir):
    """Findet die conf.py Datei für die Sphinx-Dokumentation."""
    for root, dirs, files in os.walk(base_dir):
        if 'conf.py' in files:
            return os.path.join(root, 'conf.py')
    return None

def update_breathe_config(conf_path, xml_dir):
    """Aktualisiert die Breathe-Konfiguration in der conf.py Datei."""
    if not os.path.exists(conf_path):
        print(f"Fehler: conf.py nicht gefunden unter {conf_path}")
        return False
    
    with open(conf_path, 'r') as f:
        content = f.read()
    
    # Prüfe, ob breathe konfiguriert ist
    breathe_pattern = re.compile(r'breathe_projects\s*=\s*{[^}]*}', re.DOTALL)
    breathe_default_pattern = re.compile(r'breathe_default_project\s*=\s*[\'"][^\'"]*[\'"]')
    
    # Erstelle den absoluten Pfad zum XML-Verzeichnis
    abs_xml_dir = os.path.abspath(xml_dir)
    
    # Aktualisiere oder füge die breathe_projects Konfiguration hinzu
    if breathe_pattern.search(content):
        content = breathe_pattern.sub(f'breathe_projects = {{"AnalysisG": "{abs_xml_dir}"}}', content)
    else:
        content += f'\n\n# Breathe-Konfiguration\nbreathing_projects = {{"AnalysisG": "{abs_xml_dir}"}}\n'
    
    # Aktualisiere oder füge die breathe_default_project Konfiguration hinzu
    if breathe_default_pattern.search(content):
        content = breathe_default_pattern.sub('breathe_default_project = "AnalysisG"', content)
    else:
        content += 'breathe_default_project = "AnalysisG"\n'
    
    # Stelle sicher, dass breathe in den Erweiterungen enthalten ist
    if "extensions = [" in content:
        if "'breathe'" not in content and '"breathe"' not in content:
            content = content.replace("extensions = [", "extensions = [\n    'breathe',")
    
    # Schreibe die aktualisierte Datei
    with open(conf_path, 'w') as f:
        f.write(content)
    
    print(f"Breathe-Konfiguration in {conf_path} aktualisiert")
    return True

def create_empty_index_xml(xml_dir):
    """Erstellt eine leere index.xml-Datei, wenn keine existiert."""
    index_path = os.path.join(xml_dir, "index.xml")
    if not os.path.exists(index_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<doxygenindex version="">\n')
            f.write('  <!-- Leerer Index als Fallback generiert -->\n')
            f.write('</doxygenindex>\n')
        print(f"Leere index.xml erstellt unter {index_path}")
    return os.path.exists(index_path)

def symlink_or_copy_xml_dir(source_xml_dir, target_xml_dir):
    """Erstellt einen Symlink oder kopiert Dateien, wenn die XML-Verzeichnisse unterschiedlich sind."""
    if os.path.abspath(source_xml_dir) == os.path.abspath(target_xml_dir):
        return True
    
    # Erstelle Zielverzeichnis, falls es nicht existiert
    os.makedirs(os.path.dirname(target_xml_dir), exist_ok=True)
    
    # Versuche, einen Symlink zu erstellen
    try:
        if os.path.exists(target_xml_dir):
            if os.path.islink(target_xml_dir):
                os.unlink(target_xml_dir)
            else:
                shutil.rmtree(target_xml_dir)
        
        os.symlink(source_xml_dir, target_xml_dir, target_is_directory=True)
        print(f"Symlink erstellt von {source_xml_dir} nach {target_xml_dir}")
        return True
    except Exception as e:
        print(f"Symlink konnte nicht erstellt werden: {e}")
        print(f"Kopiere stattdessen die XML-Dateien...")
        
        try:
            if os.path.exists(target_xml_dir):
                shutil.rmtree(target_xml_dir)
            shutil.copytree(source_xml_dir, target_xml_dir)
            print(f"XML-Dateien von {source_xml_dir} nach {target_xml_dir} kopiert")
            return True
        except Exception as e:
            print(f"Fehler beim Kopieren der XML-Dateien: {e}")
            return False

def main():
    """Hauptfunktion."""
    project_root = Path("/workspaces/AnalysisG")
    docs_dir = project_root / "docs"
    
    # Standardpfade
    doxygen_xml_dir = docs_dir / "doxygen" / "xml"
    expected_xml_dir = docs_dir / "docs" / "doxygen" / "xml"  # Der Pfad im Fehler
    
    # Suche nach conf.py
    conf_path = find_conf_py(docs_dir)
    if not conf_path:
        print("Fehler: Keine conf.py-Datei für Sphinx gefunden.")
        return 1
    
    # Erstelle beide Verzeichnisse, falls sie nicht existieren
    os.makedirs(doxygen_xml_dir, exist_ok=True)
    os.makedirs(expected_xml_dir, exist_ok=True)
    
    # Erstelle eine leere index.xml in beiden Verzeichnissen, falls sie nicht existieren
    create_empty_index_xml(doxygen_xml_dir)
    create_empty_index_xml(expected_xml_dir)
    
    # Erstelle einen Symlink oder kopiere Dateien zwischen den Verzeichnissen
    if os.path.exists(doxygen_xml_dir) and os.listdir(doxygen_xml_dir):
        success = symlink_or_copy_xml_dir(doxygen_xml_dir, expected_xml_dir)
    elif os.path.exists(expected_xml_dir) and os.listdir(expected_xml_dir):
        success = symlink_or_copy_xml_dir(expected_xml_dir, doxygen_xml_dir)
    else:
        print("Warnung: Beide XML-Verzeichnisse sind leer oder existieren nicht.")
        success = True  # Wir haben leere Verzeichnisse und index.xml-Dateien erstellt
    
    # Aktualisiere die Breathe-Konfiguration
    if not update_breathe_config(conf_path, expected_xml_dir):
        return 1
    
    print("Breathe-Konfiguration erfolgreich aktualisiert.")
    return 0

if __name__ == "__main__":
    sys.exit(main())