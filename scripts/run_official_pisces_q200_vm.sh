#!/usr/bin/env bash
set -euo pipefail

ROOT="${PISCES_LARGE_ROOT:-$PWD}"
DATA="${ROOT}/data_q200"
OUT="${ROOT}/results_q200"
K="${PISCES_K:-10}"
mkdir -p "${OUT}"

run_path() {
  local path="$1"
  local binary corpus query
  if [[ "${path}" == "bm25" ]]; then
    binary="${ROOT}/bin/bm25_rag_bench"
    corpus="${DATA}/bm25_corpus.jsonl"
    query="${DATA}/bm25_query.jsonl"
  else
    binary="${ROOT}/bin/similarity_rag_bench"
    corpus="${DATA}/sim_corpus.jsonl"
    query="${DATA}/sim_query.jsonl"
  fi
  echo "START ${path} $(date -u +%FT%TZ)"
  /usr/bin/time -v -o "${OUT}/${path}_fiqa_n57638_q200_k${K}.time" \
    "${binary}" \
    --database_path="${corpus}" \
    --query_path="${query}" \
    --output_file="${OUT}/${path}_fiqa_n57638_q200_k${K}.jsonl" \
    --K="${K}" \
    > "${OUT}/${path}_fiqa_n57638_q200_k${K}.log" 2>&1
  test "$(wc -l < "${OUT}/${path}_fiqa_n57638_q200_k${K}.jsonl")" -eq 200
  grep -E 'Elapsed|Maximum resident|Exit status' "${OUT}/${path}_fiqa_n57638_q200_k${K}.time"
  echo "DONE ${path} $(date -u +%FT%TZ)"
}

run_path bm25
run_path similarity
date -u +%FT%TZ > "${OUT}/official_complete.flag"
