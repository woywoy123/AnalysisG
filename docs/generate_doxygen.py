#!/usr/bin/env python3
"""
Script zur automatischen Generierung und Überprüfung der Doxygen-Dokumentation.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_and_create_directory(directory):
    """Überprüft und erstellt ein Verzeichnis, falls es nicht existiert."""
    if not os.path.exists(directory):
        print(f"Erstelle Verzeichnis: {directory}")
        os.makedirs(directory)
    return os.path.isdir(directory)

def run_doxygen(doxyfile_path, output_dir):
    """Führt Doxygen mit der angegebenen Konfigurationsdatei aus."""
    try:
        # Stelle sicher, dass das Ausgabeverzeichnis existiert
        check_and_create_directory(output_dir)
        
        # Erstelle eine temporäre Doxyfile mit dem korrekten Ausgabeverzeichnis
        temp_doxyfile = os.path.join(os.path.dirname(doxyfile_path), "Doxyfile.temp")
        with open(doxyfile_path, 'r') as f:
            content = f.read()
        
        # Setze das Ausgabeverzeichnis
        content = content.replace("OUTPUT_DIRECTORY       = doxygen", 
                                 f"OUTPUT_DIRECTORY       = {output_dir}")
        
        # Stelle sicher, dass XML-Ausgabe aktiviert ist
        content = content.replace("GENERATE_XML           = NO", 
                                 "GENERATE_XML           = YES")
        
        # Schreibe die temporäre Doxyfile
        with open(temp_doxyfile, 'w') as f:
            f.write(content)
        
        # Führe Doxygen aus
        print(f"Führe Doxygen aus mit Konfigurationsdatei: {temp_doxyfile}")
        result = subprocess.run(['doxygen', temp_doxyfile], 
                               capture_output=True, text=True)
        
        # Lösche die temporäre Datei
        os.remove(temp_doxyfile)
        
        if result.returncode != 0:
            print(f"Fehler beim Ausführen von Doxygen: {result.stderr}")
            return False
        
        # Überprüfe, ob index.xml erstellt wurde
        xml_dir = os.path.join(output_dir, "xml")
        index_path = os.path.join(xml_dir, "index.xml")
        if not os.path.exists(index_path):
            print(f"Warnung: index.xml wurde nicht erstellt im Verzeichnis: {xml_dir}")
            # Versuche, ein leeres index.xml zu erstellen
            create_empty_index_xml(xml_dir)
            return False
        
        print(f"Doxygen-Dokumentation erfolgreich erstellt im Verzeichnis: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Fehler: {e}")
        return False

def create_empty_index_xml(xml_dir):
    """Erstellt eine leere index.xml-Datei als Fallback."""
    try:
        if not os.path.exists(xml_dir):
            os.makedirs(xml_dir)
        
        index_path = os.path.join(xml_dir, "index.xml")
        with open(index_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<doxygenindex>\n')
            f.write('  <!-- Generiert als Fallback -->\n')
            f.write('</doxygenindex>\n')
        
        print(f"Leere index.xml erstellt: {index_path}")
        return True
    except Exception as e:
        print(f"Fehler beim Erstellen der leeren index.xml: {e}")
        return False

def main():
    # Konfigurationen
    project_root = Path("/workspaces/AnalysisG")
    docs_dir = project_root / "docs"
    doxyfile_path = docs_dir / "Doxyfile"
    output_dir = docs_dir / "doxygen"
    
    # Überprüfe, ob die Doxyfile existiert
    if not os.path.exists(doxyfile_path):
        print(f"Fehler: Doxyfile nicht gefunden in {doxyfile_path}")
        print("Erstelle eine Standard-Doxyfile...")
        try:
            subprocess.run(['doxygen', '-g', str(doxyfile_path)], 
                          check=True, capture_output=True)
            print(f"Standard-Doxyfile erstellt: {doxyfile_path}")
        except subprocess.CalledProcessError as e:
            print(f"Fehler beim Erstellen der Standard-Doxyfile: {e}")
            return 1
    
    # Führe Doxygen aus
    success = run_doxygen(doxyfile_path, output_dir)
    
    # Überprüfe die Ausgabe
    xml_dir = output_dir / "xml"
    if not os.path.exists(xml_dir / "index.xml"):
        print(f"Warnung: index.xml nicht gefunden nach der Doxygen-Ausführung")
        print(f"Erstelle leere index.xml als Fallback")
        create_empty_index_xml(str(xml_dir))
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())