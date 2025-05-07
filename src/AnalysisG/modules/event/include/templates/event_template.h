/** \file event_template.h
 *  \brief Header file for the event_template class.
 *
 *  This file contains the declaration of the event_template class, which is a base template class for event data representation and manipulation.
 */

/**
 * @file event_template.h
 * @brief Definiert die Basisvorlage-Klasse für Ereignisdaten-Darstellung und -Manipulation.
 *
 * Diese Datei enthält die Deklaration der `event_template`-Klasse, die als Basisvorlage
 * für die Verarbeitung von Physik-Ereignisdaten dient. Sie bietet Funktionalitäten für die Verwaltung von Trees,
 * Branches und Leaves in Physik-Datenstrukturen, sowie Eigenschaften wie Ereignisgewicht
 * und Index. Die Klasse unterstützt auch die dynamische Registrierung von Partikeln und den Aufbau von Ereignissen aus Rohdaten.
 * 
 * Die Ereignisarchitektur ist flexibel und erweiterbar gestaltet und ermöglicht die Anpassung
 * an verschiedene Arten von Physikanalysen und Datenformate.
 */

#ifndef EVENT_TEMPLATE_H ///< Start des Include-Guards für EVENT_TEMPLATE_H, um Mehrfachinklusionen zu verhindern.
#define EVENT_TEMPLATE_H ///< Definition von EVENT_TEMPLATE_H, um anzuzeigen, dass der Header inkludiert wurde.

#include <templates/particle_template.h> ///< Inkludiert die `particle_template`-Klasse für die Partikel-Datenverarbeitung.
#include <structs/property.h> ///< Inkludiert die `cproperty`-Vorlage für die benutzerdefinierte Eigenschaftsverwaltung.
#include <structs/element.h> ///< Inkludiert die `element_t`-Struktur, wahrscheinlich ein Basisdatenelement-Typ.
#include <structs/event.h> ///< Inkludiert die `event_t`-Struktur für die Ereignisdatendarstellung.
#include <tools/tools.h> ///< Inkludiert die `tools`-Klasse für Hilfsfunktionen.
#include <meta/meta.h> ///< Inkludiert die `meta`-Klasse für Metadaten-Handling.

/**
 * @class event_template
 * @brief Basisvorlage-Klasse für Ereignisdaten-Darstellung und -Manipulation.
 *
 * Erbt von `tools`, um Hilfsfunktionen bereitzustellen. Diese Klasse ist dazu gedacht, für
 * spezifische Ereignistypen oder Analysen als Unterklasse verwendet zu werden. Sie verwaltet Sammlungen von Trees, Branches und Leaves,
 * die die Ereignisstruktur definieren, sowie Eigenschaften wie Ereignisgewicht und Index.
 * Sie unterstützt die dynamische Partikelregistrierung und den Aufbau von Ereignissen aus Rohdaten.
 * 
 * Die Klasse dient als zentrale Komponente im AnalysisG-Framework und verbindet Rohdaten
 * mit Analyseobjekten und -algorithmen auf höherer Ebene. Sie bietet Schnittstellen für den Datenzugriff,
 * die Filterung und Transformation, die für Physikanalysen benötigt werden.
 */
class event_template: public tools // Definiert die Klasse 'event_template', die von 'tools' erbt.
{
public:
    /**
     * @brief Standard-Konstruktor für event_template.
     * 
     * Initialisiert ein leeres Ereignis-Template mit Standard-Eigenschaften und Einstellungen.
     * Richtet Eigenschafts-Mappings ein und initialisiert Sammlungen.
     */
    event_template();
    
    /**
     * @brief Destruktor für event_template.
     * 
     * Bereinigt zugewiesene Ressourcen, einschließlich registrierter Partikel und Datenstrukturen.
     */
    virtual ~event_template();
    
    /**
     * @brief Erstellt eine neue Ereignis-Template-Instanz.
     * @return Zeiger auf das neu erstellte event_template-Objekt.
     * 
     * Factory-Methode, die eine neue Instanz der event_template-Klasse erstellt und zurückgibt.
     * Nützlich für die Erstellung von abgeleiteten Klasseninstanzen in einem polymorphen Kontext.
     */
    virtual event_template* clone();
    
    /**
     * @brief Virtuelle Methode zum Zurücksetzen des Ereignis-Templates auf seinen Anfangszustand.
     *
     * Löscht alle Ereignisdaten und bereitet die Vorlage für die neue Ereignisverarbeitung vor.
     */
    virtual void reset();
    
    /**
     * @brief Initialisiert das Ereignis-Template mit notwendigen Einstellungen.
     *
     * Richtet interne Datenstrukturen und Konfigurationen ein. Diese Methode sollte aufgerufen werden,
     * bevor andere Operationen auf dem Template durchgeführt werden.
     */
    void initialize();
    
    /**
     * @brief Baut ein Ereignis aus Rohdaten auf.
     *
     * Virtuelle Methode, die Rohdaten verarbeitet, um eine strukturierte Ereignisrepräsentation zu erstellen.
     * Diese Basismethode leitet die Implementierung an die überladene build(element_t*) Methode weiter.
     */
    void build();
    
    /**
     * @brief Baut eine Ereignisdatenstruktur aus einem Element auf.
     * @param el Zeiger auf eine element_t-Struktur, die Rohdaten des Ereignisses enthält.
     *
     * Virtuelle Methode, die Rohelement-Daten verarbeitet, um die Ereignisstruktur aufzubauen.
     * Diese Methode wird typischerweise in abgeleiteten Klassen überschrieben, um spezifische Ereignisformate zu handhaben.
     */
    virtual void build(element_t* el);
    
    /**
     * @brief Baut ein Mapping zwischen Ereignisdatenstrukturen und Handlern auf.
     * @param evnt Zeiger auf eine Map, die Strings mit data_t-Zeigern verbindet.
     *
     * Erstellt Beziehungen zwischen den Trees, Branches, Leaves des Ereignisses und den 
     * entsprechenden Datenhandlern, um den Datenzugriff und die Manipulation zu erleichtern.
     */
    void build_mapping(std::map<std::string, data_t*>* evnt);
    
    /**
     * @brief Löscht alle Leaf-String-Referenzen.
     *
     * Setzt interne Sammlungen von Trees, Branches und Leaves zurück.
     * Dies ist nützlich bei der Vorbereitung zur Verarbeitung eines neuen Ereignisformats.
     */
    void clear_leaves();
    
    /**
     * @brief Registriert ein Partikel zur Nachverfolgung in diesem Ereignis.
     * @param name Der Name, der dieses Partikel identifiziert.
     * @param collection Der Typ oder die Sammlung, zu der dieses Partikel gehört (z.B. "Electrons").
     */
    void register_particle(std::string name, std::string collection);
    
    /**
     * @brief Fügt einen Tree zur Ereignisstruktur hinzu.
     * @param key Der Primärschlüssel oder Name für diesen Tree.
     * @param tree Der spezifische Tree-Name, wenn er sich vom Schlüssel unterscheidet, oder ein Alias.
     */
    void add_tree(std::string key, std::string tree = "");
    
    /**
     * @brief Fügt einen Branch zur Ereignisstruktur hinzu.
     * @param key Der Primärschlüssel oder Name für diesen Branch (z.B. "Electrons").
     * @param branch Der spezifische Branch-Name, wenn er sich vom Schlüssel unterscheidet, oder ein Alias.
     */
    void add_branch(std::string key, std::string branch = "");
    
    /**
     * @brief Fügt ein Leaf (Variable) zur Ereignisstruktur hinzu.
     * @param key Der Primärschlüssel oder Name für dieses Leaf (z.B. "Electrons.pt").
     * @param leaf Der spezifische Leaf-Name, wenn er sich vom Schlüssel unterscheidet, oder ein Alias.
     */
    void add_leaf(std::string key, std::string leaf = "");

    // Eigenschaften mit Property-Mapping
    
    /**
     * @brief Eigenschaft: Ein Name für dieses Ereignis-Template oder diese Instanz.
     * 
     * Diese Eigenschaft ermöglicht das Setzen und Abrufen eines benutzerdefinierten Namens für das Ereignis,
     * was bei der Identifikation oder Kennzeichnung von Ereignissen in komplexen Analysen hilfreich sein kann.
     */
    cproperty<std::string, event_template> name;
    /** 
     * @brief Statischer Setter für die `name`-Eigenschaft.
     * @param[in] name Zeiger auf einen String, der den Namen enthält.
     * @param[in] ev Zeiger auf die `event_template`-Instanz.
     */
    void static set_name(std::string*, event_template*);

    /**
     * @brief Eigenschaft: Ein Hash-String, der möglicherweise die Konfiguration oder Quelle dieses Ereignisses identifiziert.
     * 
     * Diese Eigenschaft speichert einen Hash-Wert, der typischerweise zur eindeutigen Identifizierung des Ereignisses
     * oder seiner Konfiguration verwendet wird. Dies kann für Caching oder Querverweise nützlich sein.
     */
    cproperty<std::string, event_template> hash;
    /** 
     * @brief Statischer Setter für die `hash`-Eigenschaft.
     * @param[in] hash Zeiger auf einen String, der den Hash enthält.
     * @param[in] ev Zeiger auf die `event_template`-Instanz.
     */
    void static set_hash(std::string*, event_template*);

    /**
     * @brief Eigenschaft: Der primäre TTree-Name für dieses Ereignis.
     * 
     * Diese Eigenschaft speichert den Namen des Haupt-TTrees, aus dem dieses Ereignis stammt.
     * Dies ist besonders relevant, wenn mit ROOT-Dateien gearbeitet wird, die mehrere TTrees enthalten.
     */
    cproperty<std::string, event_template> tree;
    /** 
     * @brief Statischer Getter für die `tree`-Eigenschaft.
     * @param[out] name Zeiger auf einen String zum Speichern des Tree-Namens.
     * @param[in] ev Zeiger auf die `event_template`-Instanz.
     */
    void static get_tree(std::string*, event_template*);

    /**
     * @brief Eigenschaft: Das Ereignisgewicht, das für Normalisierung oder Skalierung verwendet wird.
     * 
     * Diese Eigenschaft speichert das Gewicht des Ereignisses, welches typischerweise für die Normalisierung,
     * Cross-Section-Anpassungen oder andere physikbasierte Skalierungen verwendet wird.
     */
    cproperty<double, event_template> weight;
    /** 
     * @brief Statischer Setter für die `weight`-Eigenschaft.
     * @param[in] val Zeiger auf einen double-Wert, der den Gewichtswert enthält.
     * @param[in] ev Zeiger auf die `event_template`-Instanz.
     */
    void static set_weight(double*, event_template*);

    /**
     * @brief Eigenschaft: Der Ereignisindex oder die Eintragsnummer im Quell-TTree.
     * 
     * Diese Eigenschaft speichert die Position oder den Index des Ereignisses innerhalb der Quelldaten,
     * was für die Rückverfolgung oder Identifizierung einzelner Ereignisse nützlich ist.
     */
    cproperty<long, event_template> index;
    /** 
     * @brief Statischer Setter für die `index`-Eigenschaft.
     * @param[in] val Zeiger auf einen long-Wert, der den Indexwert enthält.
     * @param[in] ev Zeiger auf die `event_template`-Instanz.
     */
    void static set_index(long*, event_template*);

    /**
     * @brief Eigenschaft: Liste der TLeaf-Namen (Variablen), die für dieses Ereignis gelesen werden sollen.
     * 
     * Diese Eigenschaft speichert die Namen aller Variablen oder Datenfelder, die aus den Rohdaten
     * für dieses Ereignis extrahiert und verfügbar gemacht werden sollen.
     */
    cproperty<std::vector<std::string>, event_template> leaves;
    /** 
     * @brief Statischer Getter für die `leaves`-Eigenschaft.
     * @param[out] inpt Zeiger auf einen Vektor von Strings zum Speichern der Leaf-Namen.
     * @param[in] ev Zeiger auf die `event_template`-Instanz.
     */
    void static get_leaves(std::vector<std::string>*, event_template*);

    /**
     * @brief Interne Map zur Speicherung von Tree-Namen-Mappings oder Aliasen.
     * 
     * Diese Map verbindet logische Baumnamen (als Schlüssel verwendet in der Analyse) 
    
    // Particle management structures
    
    std::map<std::string, std::map<std::string, particle_template*>*> particle_link; ///< Map of registered particles by type and name.
    std::map<std::string, particle_template*(*)()> particle_generators; ///< Map of particle generator functions by type.
};

#endif // End of include guard for EVENT_TEMPLATE_H
