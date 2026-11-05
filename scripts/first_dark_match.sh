#!/usr/bin/env bash
# First real SAR↔AIS dark-vessel match: the Sentinel-1C pass of ~2026-11-11 02:10 UTC
# (2026-11-10 18:10 PST) over the Salish Sea, against AIS recorded by planetar-ais.
#
# Run from bb's crontab (2026-11-11 08:47 and, as a retry, 2026-11-12 08:47 local), so it
# no longer depends on an open Claude session. Idempotent: once a report exists it exits.
# Local only: contacts no one, deploys nothing, starts no Runpod pods.
#
#   scripts/first_dark_match.sh            # writes runs/dark-2026-11-11/{log,report.md,...}
#   DAY=2026-05-08 NEXT=2026-05-09 PASS_ID=20260508T02 scripts/first_dark_match.sh   # dry run on an old scene
set -uo pipefail
results=""

DAY=${DAY:-2026-11-11}
NEXT=${NEXT:-2026-11-12}
PASS_ID=${PASS_ID:-20261111T0[12]}   # scene sensing start in the 01–02 h UTC window
AIS_GLOB=${AIS_GLOB:-ais-2026-11-1[01].jsonl}
WS=/home/sness/github/planetarx
SAT=$WS/planetar-sat
FUS=$WS/planetar-fusion
AIS=$WS/planetar-ais/data
CKPT=$FUS/checkpoints/stage1-dualpol-cosine/stage1_best.pt
OUT=$FUS/runs/dark-$DAY
MIN_FREE_GB=150

mkdir -p "$OUT"
exec >>"$OUT/log" 2>&1
echo "=== $(date -Is) first_dark_match"
note() { echo "$*"; command -v notify-send >/dev/null && DISPLAY=:0 notify-send "planetar dark match" "$*" 2>/dev/null; true; }
stop() { note "$*"; { echo "# Dark match $DAY — not run"; echo; echo "$*"; } >"$OUT/status.md"; exit 0; }

[ "$(date +%Y)" = 2026 ] || exit 0                    # the crontab lines would fire every year
[ -f "$OUT/report.md" ] && { echo "report exists; nothing to do"; exit 0; }

# 1. disk
free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "$MIN_FREE_GB" ] || stop "/home has ${free_gb} GB free (< ${MIN_FREE_GB} GB); stopped."

# 2. fetch (CDSE credentials live in ~/.secrets, which cron doesn't load)
# shellcheck disable=SC1090
[ -f ~/.secrets ] && . ~/.secrets
cd "$SAT" || stop "no $SAT"
[ "$(git branch --show-current)" = learned-detector ] || stop "planetar-sat is not on branch learned-detector"
# Only this pass: CDSE lists each slice twice (COG and non-COG); we use the COG ones, as the
# 2026-05-08 Task 9 run did. The pass usually spans two slices; each is matched on its own.
WIN="--aoi salish-sea --start ${DAY}T00:30 --end ${DAY}T04:00"
# shellcheck disable=SC2086
listing=$(.venv/bin/planetar-sat fetch $WIN --list-only)
echo "$listing"
scenes=$(echo "$listing" | grep -oE "S1[A-D]_IW_GRD[A-Z_0-9]*${PASS_ID}[A-Z_0-9]*_COG" | sort -u)
[ -n "$scenes" ] || stop "The ~02:10 UTC $DAY Sentinel-1 scene is not in the CDSE catalogue yet (or has no COG product); the retry will look again."
# shellcheck disable=SC2086
.venv/bin/planetar-sat fetch $WIN || stop "fetch failed (see log)"

ais_files=$(cd "$AIS" && ls -d "$AIS"/ais/$AIS_GLOB "$AIS"/ais-orca/$AIS_GLOB 2>/dev/null)
[ -n "$ais_files" ] || stop "no AIS recordings matching $AIS_GLOB"

body=""
for scene in $scenes; do
  zip=data/scenes/$scene.zip
  [ -f "$zip" ] || stop "downloaded zip $zip not found"
  unzip -o -j -d data/scenes "$zip" '*/measurement/*-vv-*.tiff' '*/measurement/*-vh-*.tiff' '*/annotation/s1*-vv-*.xml' \
    || stop "unzip failed for $zip"
  lower=$(echo "$scene" | tr 'A-Z' 'a-z' | cut -d_ -f5)   # sensing start, e.g. 20261111t021012
  vv=$(ls "$SAT"/data/scenes/s1?-iw-grd-vv-${lower}-*-cog.tiff | head -1)
  vh=$(ls "$SAT"/data/scenes/s1?-iw-grd-vh-${lower}-*-cog.tiff | head -1)
  ann=$(ls "$SAT"/data/scenes/s1?-iw-grd-vv-${lower}-*-cog.xml | head -1)
  echo "scene=$scene vv=$vv vh=$vh ann=$ann"
  [ -n "$vv" ] && [ -n "$vh" ] && [ -n "$ann" ] || stop "missing VV/VH/annotation for $scene"
  dets=$OUT/dets-$lower.jsonl
  assoc=$OUT/assoc-$lower.jsonl

  # 3. learned detection (default thresh 0.1 = Run 5); --vh explicit (find_vh misses non-COG names)
  .venv/bin/planetar-sat detect "$vv" --vh "$vh" --annotation "$ann" --detector learned --ckpt "$CKPT" >"$dets" \
    || stop "detect failed for $scene (see log)"
  ndet=$(wc -l <"$dets")
  start_ann=$(grep -oE '<startTime>[^<]+' "$ann" | head -1 | cut -d'>' -f2)
  acq=$("$FUS/.venv/bin/python" -c "
import json,datetime as dt
l=open('$dets').readline()
print(dt.datetime.fromtimestamp(int(json.loads(l)['acquired_at_ns'])/1e9, dt.timezone.utc).isoformat() if l else 'n/a')")
  echo "detections=$ndet acquired_at=$acq annotation startTime=$start_ann"

  # 4+5. association + AIS-blinding (associate prints both on stderr)
  if [ "$ndet" -gt 0 ]; then
    # shellcheck disable=SC2086
    summary=$(cd "$FUS" && .venv/bin/python -m planetar_fusion.associate --dets "$dets" --ais $ais_files \
      --annotation "$ann" --out "$assoc" 2>&1) || stop "associate failed for $scene: $summary"
    summary=$(echo "$summary" | tail -1)
  else
    summary="no detections"; : >"$assoc"
  fi
  echo "$summary"
  table=$("$FUS/.venv/bin/python" -c "
import json
rows=[json.loads(l) for l in open('$assoc') if l.strip()]
dark=[r for r in rows if r['status']=='dark']
print('| # | lat | lon | score |'); print('|---|---|---|---|')
for i,r in enumerate(dark,1):
    s=r.get('score', r.get('snr',''))
    print(f\"| {i} | {r['lat']:.5f} | {r['lon']:.5f} | {s if isinstance(s,str) else round(float(s),3)} |\")
if not dark: print('| – | none | | |')")
  body+="
## \`$scene\`

- Learned detector (\`stage1-dualpol-cosine/stage1_best.pt\`, thresh 0.1): $ndet detections
- acquired_at (detections): $acq · annotation startTime: $start_ann
- **Result:** $summary

Dark candidates (no AIS within the gate):

$table
"
  results+="$summary; "
done

{
  echo "# First SAR↔AIS dark-vessel match — $DAY"
  echo
  echo "AIS: $(for f in $ais_files; do printf '%s ' "${f#"$AIS"/}"; done)"
  echo "$body"
  echo "Files in this directory: dets-*.jsonl, assoc-*.jsonl, log."
} >"$OUT/report.md"
note "Dark match done: $results— $OUT/report.md"
