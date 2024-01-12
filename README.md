### Definition der Architektur des Convolutional Neural Netzwerks:
- Implementiere die Funktionen `__init__()` und `forward()`.
- In `__init__()`: Definiere die Architektur des CNN mit:
  - Eingabeschicht: 3 Kanäle (RGB-Bild)
  - Block 1 bis 4: Jeder enthält eine COnvolution, Batch-Normalisierung, ReLU-Aktivierung und Max-Pooling
  - Flattening-Schicht und vollständig verbundene Schichten mit Dropout und ReLU-Aktivierungen

### Optimierer
- Implementiere die Funktion `configure_optimizers()`.
- Definiere den Adam-Optimierer mit einer Lernrate von 0.01 und verwende den StepLR Lernratenplaner mit einer Schrittgröße von 1

### Training, Validierung und Test Schritt
- Implementiere `training_step()`, `validation_step()` und `test_step()`.
- Implementiere eine einzelne Iteration der Trainings-, Validierungs- und Testschleifen und protokolliere jeweils die Verluste und Genauigkeiten.

### Datenladen
- Lade den Trainings- und Testdatensatz des Früchte-Datensatzes.
- Wende Vorverarbeitung an: Umwandlung in PyTorch-Tensoren und Skalierung auf (100,100).
- Teile den Trainingsdatensatz in einen Validierungs- und einen kleineren Trainingsdatensatz auf, wobei der Validierungsdatensatz 10 % des gesamten Trainingsdatensatzes sein sollte.
