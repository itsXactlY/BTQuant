# btquant_vulkan Architektur-Dokumentation

## Überblick

btquant_vulkan ist eine High-Frequency Trading-Anwendung mit einer Vulkan-basierten GPU-Rendering-Engine. Das Projekt ist modular aufgebaut und verwendet modernes C++23 mit Vulkan für GPU-Berechnungen und Rendering, sowie ImGui/ImPlot für die Benutzeroberfläche.

## Architekturübersicht

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Anwendung     │    │   Rendering      │    │   Daten-       │
│                 │    │                  │    │   verarbeitung  │
│  main.cpp       │◄──►│  Vulkan Context  │◄──►│  Data Spine   │
│  Window Mgr     │    │  Render Pipeline │    │  Market Data  │
└─────────────────┘    │  GPU Buffers    │    │  Ring Buffer  │
                       └──────────────────┘    └─────────────────┘
                                ▲
                                │
                       ┌──────────────────┐
                       │   UI System      │
                       │                  │
                       │  UI Context      │
                       │  Widgets         │
                       └──────────────────┘
```

## Modulübersicht

### 1. Hauptanwendung (`src/main.cpp`)
- Enthält die Hauptklasse `BTQuantApplication`
- Initialisiert Vulkan, UI und Datenkomponenten
- Implementiert die Haupt-Render-Schleife
- Verwaltet Fenster- und Ereignisbehandlung

### 2. Vulkan-Kontext (`src/core/vulkan_context.*`)
- Verwaltet die Vulkan-Instanz und logisches Gerät
- Implementiert Swapchain-Management
- Verwaltet Render-Pässe und Befehls-Puffer
- Stellt Synchronisationsobjekte bereit
- Implementiert die Render-Schleife

### 3. Datenverarbeitung

#### 3.1 Data Spine (`src/data/data_spine.*`)
- Stellt Speicherabbildung für `/dev/shm/btquant_hotspine` bereit
- Liest Marktdaten im Binärformat
- Unterstützt Multisymbol-Datenzugriff
- Implementiert lockfreie SPSC-Warteschlangen

#### 3.2 Market Data (`src/data/market_data.*`)
- Definiert grundlegende Datenstrukturen (OrderBook, Trade, Candle, etc.)
- Stellt Aggregationsfunktionen bereit
- Implementiert Metrikenberechnungen (VWAP, Delta, etc.)

#### 3.3 Ring Buffer (`src/data/ring_buffer.*`)
- Thread-sichere lockfreie Implementierung
- Nutzt atomare Operationen für Thread-Sicherheit
- Unterstützt Single-Producer-Single-Consumer-Szenarien

### 4. Rendering-System

#### 4.1 GPU Buffers (`src/renderer/gpu_buffers.*`)
- Abstrahiert GPU-Puffer-Verwaltung
- Implementiert Buffer-Erstellung und -Kopie
- Stellt Speicherverwaltung für GPU-Daten bereit

#### 4.2 Render Pipeline (`src/renderer/render_pipeline.*`)
- Implementiert Vulkan-Render-Pipelines
- Verwaltet Uniform-Puffer
- Stellt Descriptor-Sets für Shader bereit
- Implementiert Render-Pass-Logik

### 5. UI-System

#### 5.1 UI Context (`src/ui/ui_context.*`)
- Integriert ImGui mit Vulkan
- Verwaltet UI-Renderzyklus
- Konfiguriert Farbschema und Stil

#### 5.2 Window Manager (`src/ui/window_manager.*`)
- Implementiert das Docking-System
- Verwaltet verschiedene UI-Fenster
- Stellt Menüs und Bedienelemente bereit

#### 5.3 Widgets (`src/widgets/`)
- Spezialisierte UI-Komponenten für Handelsdaten
- Order Book Widget: Anzeige des aktuellen Orderbuches
- DOM Widget: Depth-of-Market-Visualisierung
- Trades Widget: Anzeige der neuesten Trades
- TPO Widget: Time-Price-Opportunity-Diagramm

## Datenformat

### Hotspine-Binärformat (`/dev/shm/btquant_hotspine`)
- Header: "UQTB" Magic (4 Bytes) + Version (2 Bytes) + Symbol Count (2 Bytes)
- Daten beginnen ab Offset 0x1000
- Pro Symbol: 128-Byte-Struktur mit:
  - `bid_price`, `ask_price` (double)
  - `bid_size`, `ask_size` (double)
  - `timestamp` (uint64, Mikrosekunden)
  - `seq` (uint32)
  - `flags` (uint32)
  - Padding für 128-Byte-Ausrichtung

### Symboldefinitionen
- `/dev/shm/btquant_symbols.json` enthält Symbolinformationen
- JSON-Format mit ID, Exchange und Symbol-Namen

## Build-System

### CMake-Konfiguration
- C++23-Standard erforderlich
- FetchContent für externe Abhängigkeiten (ImGui, ImPlot)
- Automatische Suche nach Vulkan, Threads, nlohmann_json
- Unterstützung für GLFW oder SDL2 als Fenstermanager

### Abhängigkeiten
- Vulkan SDK
- GLFW3 oder SDL2
- nlohmann/json (optional)
- CMake 3.28+

## Designprinzipien

### Performance
- GPU-nur-Rendering ohne CPU-Blitting
- Lockfreie Datenstrukturen für niedrige Latenz
- Compute-Shader-Verarbeitung für Heatmaps und VPVR
- Memory-Mapped-Dateien für schnellen Datenzugriff

### Modularität
- Klare Trennung zwischen Rendering, Daten und UI
- Austauschbare Komponenten
- Schnittstellenbasierte Architektur

### Skalierbarkeit
- Multisymbol-Unterstützung
- Flexible Widget-Architektur
- Erweiterbare Render-Pipeline

## Entwicklungshinweise

### Debugging
- Verwendung von Validation Layers in der Entwicklung
- Logging für kritische Fehlerzustände
- Separate Release/Debug-Konfigurationen

### Testing
- Einheitliche Tests für Datenstrukturen
- Integrationstests für die Hauptkomponenten
- Leistungsmessung für kritische Pfade

## Deployment

### Produktionsumgebung
- Optimale Compiler-Flags (-O3, march=native)
- Entfernung von Debugging-Code
- Minimale Abhängigkeiten

### Hardware-Anforderungen
- Vulkan-fähige GPU
- Ausreichender RAM für Marktdaten
- Niedrige Latenz-Netzwerkkarte für Marktdatenzugriff

## Zukunftserweiterungen

- Unterstützung für zusätzliche Chart-Typen
- Erweiterte Heatmap-Funktionalitäten
- GPU-basierte technische Indikatoren
- Unterstützung für mehrere Datenquellen
- Erweiterte Backtesting-Funktionalitäten