#!/usr/bin/env bash
set -euo pipefail

# Run the unmodified official Pisces benchmark binaries over prefixes of one
# fixed preprocessed corpus.  Paths may be overridden for another VM layout.
ROOT="${PISCES_LARGE_ROOT:-$PWD}"
SCALES="${PISCES_SCALES:-4096 8192 16384 32768 57638}"
REPEATS="${PISCES_REPEATS:-5}"
K="${PISCES_K:-10}"

mkdir -p "${ROOT}/results"
printf 'started_utc\t%s\nscales\t%s\nrepeats\t%s\nk\t%s\n' \
  "$(date -u +%FT%TZ)" "${SCALES}" "${REPEATS}" "${K}" \
  > "${ROOT}/results/scale_run_manifest.tsv"

valid_run() {
  local output="$1"
  local timing="$2"
  [[ -s "${output}" && -s "${timing}" ]] \
    && [[ "$(wc -l < "${output}")" -eq 3 ]] \
    && grep -q $'Exit status: 0' "${timing}"
}

run_one() {
  local path="$1"
  local n="$2"
  local rep="$3"
  local binary corpus query prefix
  if [[ "${path}" == "bm25" ]]; then
    binary="${ROOT}/bin/bm25_rag_bench"
    corpus="${ROOT}/data/active_bm25_corpus.jsonl"
    query="${ROOT}/data/bm25_query.jsonl"
    prefix="bm25"
  else
    binary="${ROOT}/bin/similarity_rag_bench"
    corpus="${ROOT}/data/active_sim_corpus.jsonl"
    query="${ROOT}/data/sim_query.jsonl"
    prefix="similarity"
  fi

  local stem="${ROOT}/results/${prefix}_fiqa_n${n}_q3_k${K}_rep${rep}"
  if valid_run "${stem}.jsonl" "${stem}.time"; then
    echo "SKIP valid ${prefix} n=${n} rep=${rep}"
    return
  fi
  echo "RUN ${prefix} n=${n} rep=${rep} $(date -u +%FT%TZ)"
  /usr/bin/time -v -o "${stem}.time" \
    "${binary}" \
    --database_path="${corpus}" \
    --query_path="${query}" \
    --output_file="${stem}.jsonl" \
    --K="${K}" \
    > "${stem}.log" 2>&1
  grep -E 'Elapsed|Maximum resident|Exit status' "${stem}.time"
}

for n in ${SCALES}; do
  echo "PREPARE n=${n}"
  head -n "${n}" "${ROOT}/data/bm25_corpus.jsonl" > "${ROOT}/data/active_bm25_corpus.jsonl"
  head -n "${n}" "${ROOT}/data/sim_corpus.jsonl" > "${ROOT}/data/active_sim_corpus.jsonl"
  for rep in $(seq 1 "${REPEATS}"); do
    run_one bm25 "${n}" "${rep}"
    run_one similarity "${n}" "${rep}"
  done
done

printf 'finished_utc\t%s\n' "$(date -u +%FT%TZ)" >> "${ROOT}/results/scale_run_manifest.tsv"
echo "ALL REQUESTED RUNS COMPLETE"
