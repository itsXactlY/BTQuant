#!/bin/bash
# =============================================================================
#  sync_venv.sh  --  dependencies/backtrader/ in die .btq-Venv spiegeln
#
#  LIEGT IM REPO-ROOT, WEIL MAN ES SONST VERGISST. 2026-09-28 hat es einen
#  kompletten Debug-Zyklus gekostet, bis klar war, dass hier zwei Kopien
#  existieren und nur eine davon laeuft.
#
#  -----------------------------------------------------------------------------
#  DIE FALLE
#
#  Der echte Einstieg ist:   python3.14 <Skript>.py   (aus dem Repo-Root)
#  Dort ist sys.path[0] = '' bzw. das Repo-Root -- NICHT dependencies/.
#
#      sys.path.insert(0, '/home/alca/projects/PubBTQuant')   # dependencies/bt/utils/backtest.py
#      import backtrader
#
#  Also laedt der echte Lauf:
#      /home/alca/.btq/lib/python3.14/site-packages/backtrader/
#  und NICHT  dependencies/backtrader/.
#
#  Folge: jeder Fix in dependencies/backtrader/ ist ein No-op fuer den echten
#  Lauf, bis diese Kopie hier durchgerutscht ist. Du benchmarkst und profilest
#  dann den ALTEN Code und schliesst daraus, der Fix habe nicht geholfen.
#
#  Die Venv hat kein pip. Der Weg ist rsync, nicht pip install.
#
#  -----------------------------------------------------------------------------
#  AUFRUF
#      ./sync_venv.sh            spiegeln + pruefen
#      ./sync_venv.sh --check    nur pruefen, nichts schreiben (Exit 1 bei Drift)
# =============================================================================
set -u

# --- Ort aus der Position dieses Skripts ableiten, nicht aus dem CWD --------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$REPO/dependencies/backtrader"
DST=/home/alca/.btq/lib/python3.14/site-packages/backtrader
PY=/home/alca/.btq/bin/python3.14

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

RED=$'\033[1;31m'; GREEN=$'\033[1;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'

for d in "$SRC" "$DST"; do
  [ -d "$d" ] || { echo -e "${RED}FEHLER: $d existiert nicht${NC}" >&2; exit 2; }
done
[ -x "$PY" ] || { echo -e "${RED}FEHLER: $PY nicht ausfuehrbar${NC}" >&2; exit 2; }

# --- Spiegeln ----------------------------------------------------------------
if [ "$CHECK_ONLY" -eq 0 ]; then
  echo -e "${YELLOW}>>> rsync $SRC -> $DST${NC}"
  rsync -a --delete --exclude='*.pyc' "$SRC/" "$DST/" || exit 2
  # Stale Bytecode-Caches koennen eine Aenderung maskieren.
  find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
else
  echo -e "${YELLOW}>>> --check: keine Aenderung, nur Pruefung${NC}"
fi

SYNCED=1
echo

# --- Pruefung 1: sind die Kopien bytegleich? --------------------------------
echo ">>> [1/3] Hash-Vergleich aller .py"
# 'btq' ist ein verwaistes CLI-Skript (imports .btquant, existiert nicht),
# kein Modul -> kein Sync-Ziel, ausgenommen.
DIFFS=$(diff -rq -x '__pycache__' -x 'btq' "$SRC" "$DST" 2>/dev/null)
if [ -z "$DIFFS" ]; then
  echo "      OK  $(find "$SRC" -name '*.py' | wc -l) Dateien identisch"
else
  echo "$DIFFS" | sed 's/^/      /'
  echo -e "      ${RED}FEHLER: Kopien weichen ab${NC}"
  SYNCED=0
fi
echo

# --- Pruefung 2: welches backtrader laedt der echte Aufruf wirklich? -------
echo ">>> [2/3] Welche Kopie wird importiert?"
LOADED=$("$PY" -c "import backtrader; print(backtrader.__file__)" 2>/dev/null | tail -1)
echo "      $LOADED"
case "$LOADED" in
  "$DST"/*) echo "      -> Venv-Kopie. Das ist der Code, der wirklich laeuft." ;;
  *)        echo -e "      ${RED}-> WARNUNG: etwas anderes laedt zuerst!${NC}"; SYNCED=0 ;;
esac
echo

# --- Pruefung 3: ist der geladene Helfer wirklich der gepatchte? -------------
echo ">>> [3/3] Ist der geladene Code gepatcht?"
RB=$("$PY" -c "
import importlib, inspect, re
m = importlib.import_module('backtrader.utils.backtest')
r = re.search(r'runonce\s*=\s*(\w+)', inspect.getsource(m.backtest))
print('runonce=' + (r.group(1) if r else 'NICHT_GEFUNDEN'))
" 2>/dev/null | tail -1)
echo "      backtrader.utils.backtest: $RB"
if [ "$RB" = "runonce=True" ]; then
  echo "      -> gepatchter Code aktiv."
else
  echo -e "      ${RED}-> der geladene Code ist NICHT der gepatchte${NC}"
  SYNCED=0
fi

# --- Ergebnis ----------------------------------------------------------------
echo
if [ "$SYNCED" -eq 1 ]; then
  echo -e "${GREEN}ERGEBNIS: synchron. Aenderungen an dependencies/backtrader/ wirken sofort.${NC}"
  exit 0
else
  echo -e "${RED}ERGEBNIS: NICHT synchron. Fixes in dependencies/ kommen NICHT an.${NC}"
  echo -e "${RED}  Jede Messung von eben galt dem FALSCHEN Code. Erst ./sync_venv.sh, dann neu messen.${NC}"
  exit 1
fi
