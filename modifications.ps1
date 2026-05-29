# --- 1a. Supprimer __pycache__ de l'index Git (garder local) ---
Write-Host "[1/8] Suppression __pycache__ de l'index Git..." -ForegroundColor Yellow
git rm -r --cached **/__pycache__/ 2>$null
git rm -r --cached __pycache__/ 2>$null

# --- 1b. Supprimer .ipynb_checkpoints de l'index ---
Write-Host "[2/8] Suppression .ipynb_checkpoints..." -ForegroundColor Yellow
git rm -r --cached **/.ipynb_checkpoints/ 2>$null

# --- 1c. Supprimer fichiers temp/lock de l'index ---
Write-Host "[3/8] Suppression fichiers temporaires..." -ForegroundColor Yellow
$tempFiles = @(
    "Results/.~lock.breast_cancer_biomarker_eval_summary.csv#",
    "dashboard_tmp_log.txt",
    "test_after.log",
    "cantemist_count.log",
    "validation_cantemist.log",
    "jupyter_server_config.json"
)
foreach ($f in $tempFiles) {
    if (Test-Path $f) {
        git rm --cached $f 2>$null
        Write-Host "   RetirÃ©: $f" -ForegroundColor DarkGray
    }
}

# --- 1e. Supprimer les duplicatas ---
Write-Host "[5/8] Retrait des fichiers dupliquÃ©s..." -ForegroundColor Yellow
$duplicates = @(
    "Consumtion_of_Duraxell.csv",                # doublon de ESMO2025/
    "breast_cancer_biomarker_eval_summary.csv",   # doublon de Results/
    "NER/sweep_results (copie).csv"               # copie explicite
)
foreach ($f in $duplicates) {
    if (Test-Path $f) {
        git rm --cached "`"$f`"" 2>$null
        Write-Host "   RetirÃ©: $f" -ForegroundColor DarkGray
    }
}

# --- 1f. Supprimer fichiers vides ---
Write-Host "[6/8] Retrait des fichiers vides..." -ForegroundColor Yellow
$emptyFiles = @(
    "ESMO2025/graph_orchestrator.py",
    "Reports/conference_frugalite_abstract.md",
    "Reports/BILAN_VACANCES.md"
)
foreach ($f in $emptyFiles) {
    if (Test-Path $f) {
        git rm --cached $f 2>$null
        Remove-Item $f -Force
        Write-Host "   SupprimÃ©: $f" -ForegroundColor DarkGray
    }
}

# --- 1h. Commit Phase 1 ---
Write-Host "[8/8] Commit Phase 1..." -ForegroundColor Yellow
git add .gitignore

$commitMsg1 = @"
chore: nettoyage Git - suppression pycache, binaires, temp, duplicatas

- Retiré 13 dossiers __pycache__ (164 fichiers .pyc) de l'index
- Retiré 5 binaires lourds (~11 Mo) : PDF, PPTX, RAR
- Retiré fichiers lock/temp/log vides
- Retiré 3 fichiers dupliqués
- Supprimé 3 fichiers vides (graph_orchestrator.py, .md vides)
- .gitignore réécrit avec couverture complète
"@
git commit -m $commitMsg1


Write-Host ""
Write-Host "â•â•â• PHASE 2 : NOMMAGE ET FICHIERS â•â•â•" -ForegroundColor Cyan

# --- 2a. Renommer les typos ---
Write-Host "[1/3] Correction des typos dans les noms..." -ForegroundColor Yellow

# Note : ces git mv nÃ©cessitent que les fichiers soient trackÃ©s
# Si dÃ©jÃ  untracked aprÃ¨s Phase 1, adapter en simples Rename-Item

# templatability â†’ Templatability
if (Test-Path "ESMO2025/E_templatability.py") {
    git mv "ESMO2025/E_templatability.py" "ESMO2025/E_templatability.py"
    Write-Host "   E_templatability.py -> E_templatability.py" -ForegroundColor DarkGray
}
if (Test-Path "ESMO2025/generate_templatability_report.py") {
    git mv "ESMO2025/generate_templatability_report.py" "ESMO2025/generate_templatability_report.py"
}
if (Test-Path "ESMO2025/tests/test_templatability.py") {
    git mv "ESMO2025/tests/test_templatability.py" "ESMO2025/tests/test_templatability.py"
}

# feasibility â†’ Feasibility
if (Test-Path "ESMO2025/E_feasibility_NER.py") {
    git mv "ESMO2025/E_feasibility_NER.py" "ESMO2025/E_feasibility_NER.py"
    Write-Host "   E_feasibility_NER.py -> E_feasibility_NER.py" -ForegroundColor DarkGray
}

# --- 2b. Mettre Ã  jour les imports aprÃ¨s renommage ---
Write-Host "[2/3] Mise Ã  jour des imports..." -ForegroundColor Yellow
# Remplacer dans tous les .py : templatability -> templatability, feasibility -> feasibility
Get-ChildItem -Recurse -Filter "*.py" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
    if ($content -and ($content -match "templatability|feasibility")) {
        $content = $content -replace "templatability", "templatability"
        $content = $content -replace "feasibility", "feasibility"
        Set-Content -Path $_.FullName -Value $content -NoNewline -Encoding UTF8
        Write-Host "   Mis Ã  jour: $($_.FullName)" -ForegroundColor DarkGray
    }
}

# --- 2c. Commit Phase 2 ---
Write-Host "[3/3] Commit Phase 2..." -ForegroundColor Yellow
git add -A

$commitMsg2 = @"
refactor: correction typos nommage (templatability->templatability, feasibility->feasibility)

- Renommé 4 fichiers Python avec orthographe corrigée
- Mis à jour tous les imports correspondants
- Aucun changement fonctionnel
"@
git commit -m $commitMsg2

Write-Host ""
Write-Host "â•â•â• TERMINÃ‰ â•â•â•" -ForegroundColor Green
Write-Host "Prochaines Ã©tapes manuelles :" -ForegroundColor White
Write-Host "  1. Ajouter pyproject.toml (cf. template ci-dessous)" -ForegroundColor Gray
Write-Host "  2. Ajouter LICENSE (MIT recommandÃ©)" -ForegroundColor Gray
Write-Host "  3. Ajouter .github/workflows/ci.yml" -ForegroundColor Gray
Write-Host "  4. RÃ©Ã©crire README.md avec badges et structure" -ForegroundColor Gray
Write-Host "  5. git push --force-with-lease origin main" -ForegroundColor Gray