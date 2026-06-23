# 1. Initiale Befehle: SetUp

## 1.1 Virtual Environment (.venv) erstellen
Falls die Umgebung neu erstellt werden muss (Windows-Fix für Anaconda):
```powershell
& "C:\Users\Patrick Moj\anaconda3\python.exe" -m venv .venv
```

## 1.2 Installieren von neuen Paketen (gezielter pip-Zugriff)
Falls Packages installiert werden müssen:
```powershell
.\.venv\Scripts\pip install torch numpy scipy
.\.venv\Scripts\pip install torch==2.2.1 numpy==1.26.4 scipy==1.12.0
```
## 1.3 Einmalige Nutzer-Einrichtung: 
Nutzername & Mail in GitHub:
```powershell
git config --global user.name "patrickmj03"
git config --global user.email "patrick.moj@tu-berlin.de"
```

# 2. Herstellen von Verbindung zum GitHub Repo:

## 2.1 Einmalige initaiel Verbindung zur Cloud: 
Diese Schritte beim erstmaligen Verbinden beachten: 
```powershell
git remote add origin https://github.com/patrickmj03/pinn_code.git
git push -u origin main
```
## 2.2 Daily Work-Flow: 
Änderungen aus der Cloud holen (Start der Arbeit):
```powershell
git pull
```

Hochladen des Zwischenstandes in die Cloud (Abschluss der Arbeit):
```powershell
git add .
git commit -m "Zwischenstand vom $(Get-Date -Format 'dd.MM.yyyy HH:mm')"
git push
```

## 2.3 Zugriff auf Zwischen-Speicherungen: 
Verlauf der Speicherstände öffnen (liefert ID für Folge-Befehl:):
```powershell
git log --oneline
```

Sprung in die Vergangenheit (nachschauen, wie Code einmal aussah):
```powershell
git checkout <HASH-ID>
git checkout main
```

Harter Reset auf gegebenen Zwischenstand (Rest in uwiderruflich verloren): 
```powershell
git reset --hard <HASH-ID>
git push --force origin main
```

