# PINN Projekt - Wichtige Befehle

## 1. Virtual Environment (.venv) erstellen
Falls die Umgebung neu erstellt werden muss (Windows-Fix für Anaconda):
```powershell
& "C:\Users\Patrick Moj\anaconda3\python.exe" -m venv .venv
```

## 2. Installieren von neuen Paketen (ohne pip)
Falls Packages installiert werden müssen ohne pip zu nutzen: 
```powershell
.\.venv\Scripts\pip install torch numpy scipy
```

```powershell
.\.venv\Scripts\pip install torch==2.2.1 numpy==1.26.4 scipy==1.12.0
```