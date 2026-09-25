#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$HOME/duetdpe_baselines/Pisces}"
cache_root="$repo_root/distdir/content_addressable/sha256"
mkdir -p "$cache_root"

fetch_one() {
  local label="$1"
  local expected_spec="$2"
  local upstream="$3"
  shift 3
  local expected
  if [[ "$expected_spec" == hex:* ]]; then
    expected="${expected_spec#hex:}"
  else
    expected="$(printf '%s' "$expected_spec" | base64 -d | xxd -p -c 256)"
  fi

  local cache_dir="$cache_root/$expected"
  local final_path="$cache_dir/file"
  local archive_name="${upstream##*/}"
  local distfile_dir="$repo_root/experiment_logs/distfiles/$expected"
  materialize_distfile() {
    mkdir -p "$distfile_dir"
    ln -f "$final_path" "$distfile_dir/$archive_name"
  }
  if [[ -f "$final_path" ]] && [[ "$(sha256sum "$final_path" | cut -d' ' -f1)" == "$expected" ]]; then
    materialize_distfile
    printf 'CACHED %s %s\n' "$label" "$expected"
    return
  fi

  mkdir -p "$cache_dir"
  local tmp_path="$cache_dir/file.part"
  rm -f "$tmp_path"

  local mirror="${upstream/https:\/\/github.com\//https:\/\/mirror.bazel.build\/github.com\/}"
  local codeload=""
  if [[ "$upstream" =~ ^https://github.com/([^/]+)/([^/]+)/archive/(.+)\.tar\.gz$ ]]; then
    codeload="https://codeload.github.com/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}/tar.gz/${BASH_REMATCH[3]}"
  fi

  local candidate extra
  local ok=0
  local -a candidates=("$mirror")
  # GitHub archive URLs redirect to codeload, which is reachable from the VM
  # even when github.com itself is rejected by the transparent network layer.
  [[ -n "$codeload" ]] && candidates+=("$codeload")
  for extra in "$@"; do
    candidates+=("$extra")
  done
  candidates+=("$upstream")
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    printf 'FETCH %s %s\n' "$label" "$candidate"
    if curl -fL --retry 6 --retry-connrefused --retry-delay 2 --connect-timeout 15 \
      -o "$tmp_path" "$candidate"; then
      if [[ "$(sha256sum "$tmp_path" | cut -d' ' -f1)" == "$expected" ]]; then
        ok=1
        break
      fi
      printf 'CHECKSUM_MISMATCH %s %s\n' "$label" "$candidate" >&2
    fi
    rm -f "$tmp_path"
  done

  if [[ "$ok" -ne 1 ]]; then
    printf 'FAILED %s\n' "$label" >&2
    exit 1
  fi
  mv "$tmp_path" "$final_path"
  materialize_distfile
  printf 'VERIFIED %s %s %s\n' "$label" "$expected" "$(stat -c %s "$final_path")"
}

# Values come from the registries pinned in MODULE.bazel.lock. The final two
# extension repositories expose hexadecimal sha256 values directly in the lock.
fetch_one abseil-cpp '9Q5awxGoE4Laf6dblzEOS5AGR0+VYKxG9UqZZ/B9SuM=' 'https://github.com/abseil/abseil-cpp/releases/download/20240722.0/abseil-cpp-20240722.0.tar.gz'
fetch_one blake3 '3dJPJqMdIzc+Y9m+LnIyY6xGyLbUmQKrCAJLVz/SpBY=' 'https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.5.4.tar.gz'
fetch_one boost-multiprecision 'rJgUVSZlBR+lKCBtjT+EFQFeKrcyIOHXwCd7qwO9bT4=' 'https://github.com/boostorg/multiprecision/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-sort '8TGxMuHFerMfYe1BvZeEN6fRQuxwYfwrBX3gVnBwMvA=' 'https://github.com/boostorg/sort/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one fmt 'QPxYvrzzjHWeEae9j9wWNQfSQj71BYu6fyYoDFucVGU=' 'https://github.com/fmtlib/fmt/releases/download/11.0.2/fmt-11.0.2.zip' 'https://sources.buildroot.net/fmt/fmt-11.0.2.zip'
fetch_one msgpack-c 'I+3n6TyO/uNDrYxlFMKPNwggflEGrzs+SWmzqe1wOec=' 'https://github.com/msgpack/msgpack-c/releases/download/cpp-6.1.0/msgpack-cxx-6.1.0.tar.gz' 'https://security.ubuntu.com/ubuntu/pool/universe/m/msgpack-cxx/msgpack-cxx_6.1.0.orig.tar.gz'
fetch_one nlohmann-json 'uMsO8t1/V/GJM5l8mTS7H6liWU9wHNWo08LIBUFVk3I=' 'https://github.com/nlohmann/json/releases/download/v3.12.0/include.zip' 'https://downloads.sourceforge.net/project/json-for-modern-c.mirror/v3.12.0/include.zip'
fetch_one rules-proto 'DlxkolmabibGoD1hYiQtIx7MDeIZU0w4y0QCFx3vIeg=' 'https://github.com/bazelbuild/rules_proto/releases/download/7.0.2/rules_proto-7.0.2.tar.gz'
fetch_one spdlog 'FYZQgCmn0GcN/LLZdXXc3CQtOGiiWXQrafEAgBq04Ws=' 'https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.tar.gz'
fetch_one cpu-features '7ZaS7B9P8v9z5F6nU9odLuWorq0R5GZ+kKVtEklo/Dg=' 'https://github.com/google/cpu_features/archive/7a8174a371e2253b7bd025000a65aec1a2aa93be.tar.gz'
fetch_one emp-ot 'NYA25dGBQ3IO4XED+BckR94jAUvPwfjn1YScUlypKKw=' 'https://github.com/emp-toolkit/emp-ot/archive/refs/tags/0.2.4.tar.gz'
fetch_one emp-tool 'uasjgDEueAIDRrXS2z0CRMe9gJjLUPizYgUy70kYCNA=' 'https://github.com/emp-toolkit/emp-tool/archive/refs/tags/0.2.5.tar.gz'
fetch_one fourqlib 'dBfIKdeTP6zaVox6CJJN/vsMg90dq0EeWXr0wMwEF/A=' 'https://github.com/microsoft/FourQlib/archive/1031567f23278e1135b35cc04e5d74c2ac88c029.tar.gz'
fetch_one leveldb 'mjf4phdPCb1iK8cjtViB3FQc1QdHy9CIMcKoLWIPbXY=' 'https://github.com/google/leveldb/archive/refs/tags/1.23.tar.gz'
fetch_one libsodium 'b1BEkLNCpPikxKAvybhmy++GItXfTlRStGvhIeRmNsE=' 'https://github.com/jedisct1/libsodium/releases/download/1.0.18-RELEASE/libsodium-1.0.18.tar.gz' 'https://download.libsodium.org/libsodium/releases/old/libsodium-1.0.18.tar.gz'
fetch_one libtommath 'fPvbZEMRKd5CV+fTNJIA/b1PIptHD/NBezDQ85vu1B8=' 'https://github.com/libtom/libtommath/archive/42b3fb07e7d504f61a04c7fca12e996d76a25251.tar.gz'
fetch_one openssl 'vtuxaVVVX5mxp7G6kPyXh560ECUIG+NZ7Nap/L3xyNI=' 'https://github.com/openssl/openssl/archive/refs/tags/openssl-3.3.2.tar.gz'
fetch_one seal 'r5vw8NrM2iqLfzRPE6VpLg7mpF/qiEeLK5DDVki/JnI=' 'https://github.com/microsoft/SEAL/archive/refs/tags/v4.1.1.tar.gz'
fetch_one simplest-ot 'yIFr8UfjIPUcUW9MUR8tGnMqwNDxcdKfRCy+K1Fz3bo=' 'https://github.com/secretflow/simplest-ot/archive/60197bc7dad327bb55759e8e854885411e999167.tar.gz'
fetch_one ntl 'hex:ef578fa8b6c0c64edd1183c4c303b534468b58dd3eb8df8c9a5633f984888de5' 'https://github.com/libntl/ntl/archive/refs/tags/v11.5.1.tar.gz'
fetch_one xtensor 'hex:f5f42267d850f781d71097b50567a480a82cd6875a5ec3e6238555e0ef987dc6' 'https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.26.0.tar.gz'
fetch_one xtl 'hex:ee38153b7dd0ec84cee3361f5488a4e7e6ddd26392612ac8821cbc76e740273a' 'https://github.com/xtensor-stack/xtl/archive/refs/tags/0.8.0.tar.gz'

# Transitive Boost modules. They deliberately live in separate hash-addressed
# distdirs because GitHub assigns every module the same archive basename.
fetch_one boost-assert 'HSrhT/hAiMvtdHKWyIOB3RPrV7JeAJnrjdd4MmpvpHY=' 'https://github.com/boostorg/assert/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-config 'uSfF4CxBqoM7fVxPnAnddmjnr/EIal47YlfU5IJWAbA=' 'https://github.com/boostorg/config/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-core 'gAOK9OlsjOfK1jY+8boC3FEBf9qDKKzlmKPNQ9lxOjg=' 'https://github.com/boostorg/core/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-integer 'tlh87M1ox4iHMUDzZYH06KU8w2W6qSmBjhXwI5tUvRQ=' 'https://github.com/boostorg/integer/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-lexical-cast '5QXqwGcj8JUXSnmuxm0zLzbz9MmJrm6IKgxXegt60LQ=' 'https://github.com/boostorg/lexical_cast/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-math 'U+X3U5pmiZ/g/KMIBAXL1feVnaU5TsE2ZHRnQa7OFwU=' 'https://github.com/boostorg/math/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-predef 'wNijX5JYhG+dqlEcDeBS/M4MiNvh5pfzBHEYwx+hOFQ=' 'https://github.com/boostorg/predef/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-random '5qVfaZ7Uy/d5CaYd+MBJGxqt5GxSrnYMDC3zdhkZrV8=' 'https://github.com/boostorg/random/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-range 'f3PxqvkgpskYyvcRHlCvZSCiq9IpUHydi3WLRs514s8=' 'https://github.com/boostorg/range/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-static-assert '7Zi41407qa/Fqkdyk1NL4W+oWwza4EFMSMK3xHXDhtM=' 'https://github.com/boostorg/static_assert/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-throw-exception 'VG/INzzr/W/aKroeUVJricbw/26lVtFTVGXCvMrsP1M=' 'https://github.com/boostorg/throw_exception/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-type-traits 'w7gOSDuQ8vng6bT98x7bfU1SIa7TrkxKehoU7gJXaSA=' 'https://github.com/boostorg/type_traits/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-array 'b3X1Ov5KAosAdEY0DnPZaSVAnrTNofwY04kIS3ofNiA=' 'https://github.com/boostorg/array/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-concept-check 'XYNoKdHVdNG/Abb4ESI0OsjbW3bQ0gtFvnz4HbavWpY=' 'https://github.com/boostorg/concept_check/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-container-hash '6j4lpgLEsMQsZC/cENLn4GTja2z842j5PrXyDLvZUDU=' 'https://github.com/boostorg/container_hash/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-conversion 'LcTvJVMEQpJDhbqOL6fzecbKWiMDUxwizzhdDAQo4HE=' 'https://github.com/boostorg/conversion/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-detail 'bR2mqmLf56CyWpmlciUYscTTKAq9S05hEloNTWLBuuE=' 'https://github.com/boostorg/detail/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-dynamic-bitset 'WVdk7ci9FDJinetJ/4fnRSpuQ5MKL/+kJLJUPtzyrYA=' 'https://github.com/boostorg/dynamic_bitset/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-io '2gS/1jh06gSFTT3ha8LOztJ7AcIJasoy/c49G0IVRN8=' 'https://github.com/boostorg/io/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-iterator 'fZub0GVtQlf2DuScpGzAaIIwC3CJJGY/FGzCLx7TSQU=' 'https://github.com/boostorg/iterator/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-mpl 'ckylsTYlc6p5gSHIDSPQdzGg33QcYaRvDYurMxgQPNQ=' 'https://github.com/boostorg/mpl/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-numeric-conversion '/PKczBsLK7T+8MNQD3q5TFxlbN11iqUGB+d71kd9J5c=' 'https://github.com/boostorg/numeric_conversion/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-optional 'Q2ub2unT9FSJ5zzssfsD7IQWMRD67/CgOa0esv09kDE=' 'https://github.com/boostorg/optional/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-preprocessor 'a8hKNH8rYwDETG7kjVRpErZBZvM+BNpH34McLweS8XU=' 'https://github.com/boostorg/preprocessor/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-regex 'GqnL6F6aAHeyCj2j0aSNvyuVZNHHcODMWJU1Sv+fRhY=' 'https://github.com/boostorg/regex/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-system 't0k31L8jYoUkdNU7WtIx52BJEg/ILcL2RNPxAwuQdH4=' 'https://github.com/boostorg/system/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-tuple 'TGjPfu0CwK/0u2C/aVwyNJbsNd5NS8xIXEu39eKYmXw=' 'https://github.com/boostorg/tuple/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-utility 'bhEtQjLibNdxc/y9TwCjHbz55HayK7ScepBghxrjdsM=' 'https://github.com/boostorg/utility/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-container 'DcVI5iyC2hxWpnJ0CWDAedfcO3W+f+GLohos8pIBU80=' 'https://github.com/boostorg/container/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one gflags 'NK8vFc9zZ1E7NSvc0kk6sUzkNpLS3NnfxJlJKWbGTc8=' 'https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz'
fetch_one boost-intrusive 'T96q/xIwSFzM8BJ4/PefK+kHiqHu5RPc5YzNzMB7xqQ=' 'https://github.com/boostorg/intrusive/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-move 'Y6C/bco+qpe94T8rLI+FmXwbvmloGsXhlR2VX0FUue0=' 'https://github.com/boostorg/move/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-winapi 'ZI//e/w2tWsU6R8qfLaMNd6+ETxcE5JidZSoJPe6jGU=' 'https://github.com/boostorg/winapi/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-variant2 '80QXswAcgFA4wQJQMg+9k8nKs6jtmcu2ZxWD3tpwXn8=' 'https://github.com/boostorg/variant2/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-describe 'pIPnzATQYMAUljW4HM9fYqfiqQYvXUtO52i40p2j/U8=' 'https://github.com/boostorg/describe/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-mp11 'WwpDl/84bguU5m8Yzm09vJIwrOQESLvHmjljwWbDJtE=' 'https://github.com/boostorg/mp11/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-smart-ptr 'E5t/4giJOpB5BqFXZrY0nX3e2N5qRVEo8k3xX57zrBc=' 'https://github.com/boostorg/smart_ptr/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-fusion 'cd2i+wxpBttO75Z5CL6LFJVM6YRM6enaKQ91BWiPdD0=' 'https://github.com/boostorg/fusion/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-typeof 'jIfc34VUdh7LjsUFGpGLTD2K9lkrOQftMBQ6J34kuw0=' 'https://github.com/boostorg/typeof/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-function-types 'e42k0zTpAo6pBT3ojtCjz+HQT0xI1fAsfMX31GTSwPA=' 'https://github.com/boostorg/function_types/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-functional 'DbZLDFQJx8Ko0pHwXmDBuNgVotKqK6EjNS3iNgB1VHg=' 'https://github.com/boostorg/functional/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-functional-build-overlay 'BXgTucPX1SxSP0cmChd04GOoL6NSgIS/8WvZUrDnv9M=' 'https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.functional/1.83.0/overlay/BUILD.bazel' 'https://gh-proxy.com/https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.functional/1.83.0/overlay/BUILD.bazel'
fetch_one boost-function 'AEMEbJJjtSsMYSZJgb3p/H7ImZ/wYsGH1hiy5upro/I=' 'https://github.com/boostorg/function/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-function-module-overlay '+oz4OPdiiRgCEPF+X17/tRuRzgvgosxyorE05TAGu9Y=' 'https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.function/1.83.0/overlay/MODULE.bazel' 'https://gh-proxy.com/https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.function/1.83.0/overlay/MODULE.bazel'
fetch_one boost-function-build-overlay 'X8FNRPhbavbSfmPcfCxCveaQiyv/JUupH3ovTWlEeyk=' 'https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.function/1.83.0/overlay/BUILD.bazel' 'https://gh-proxy.com/https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.function/1.83.0/overlay/BUILD.bazel'
fetch_one boost-bind 'U+HLqJ68bQ4dATYDq47bCPiy+792BrkeurgxCsSqwa8=' 'https://github.com/boostorg/bind/archive/refs/tags/boost-1.83.0.tar.gz'
fetch_one boost-bind-module-overlay 'EeXbR/kX60oCOOibwO2itBqXloLGecwhIJ29NHw/RyM=' 'https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.bind/1.83.0/overlay/MODULE.bazel' 'https://gh-proxy.com/https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.bind/1.83.0/overlay/MODULE.bazel'
fetch_one boost-bind-build-overlay 'tGcsmQc0glQgLxJ3lPnfqq4YA4ad5G76kEAPs2K6J7I=' 'https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.bind/1.83.0/overlay/BUILD.bazel' 'https://gh-proxy.com/https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/boost.bind/1.83.0/overlay/BUILD.bazel'

# Archives exposed by registry modules or Bazel module extensions.
fetch_one gsl '8OMssQZU/qka1WveiRcNeM+/Q2PuCwHY8JfeK6SfbOk=' 'https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.tar.gz'
fetch_one zlib 'mpOyt9/ax3zrpaVYpYDnRmfdb+3kWFuR7vtg8Dty3yM=' 'https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz'
fetch_one zstd 'jCngbPQqrMHq/EB3ri7Gxvy5amJhV+BZPV6Co0/UA8E=' 'https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz' 'https://sources.buildroot.net/zstd/zstd-1.5.6.tar.gz'
fetch_one brpc 'Zc78ZgerAgU3s35xI/cswiJAJEtmSp1RlO7nO2acKGc=' 'https://github.com/apache/brpc/archive/282bc902af0b0cdf920e882cf4f1277119121fad.tar.gz'
fetch_one llvm-project '8nyhvWUvgg7Yfu7ACiGLOodGkFICeGDodH6S9cuhE5E=' 'https://github.com/llvm/llvm-project/archive/35f55f53dfbb62902da007f308a618192102dd1c.tar.gz'
fetch_one interconnection 'ozOaOj5//vqy9CwBGxJ3kNDXW3QNKtBadakgE9hA4Gw=' 'https://github.com/secretflow/interconnection/archive/b9dce7ecc901639ea38fa22e45ab0b18e8eb7787.tar.gz'
fetch_one cmake-linux-x86-64 'hex:14e15d0b445dbeac686acc13fe13b3135e8307f69ccf4c5c91403996ce5aa2d4' 'https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-x86_64.tar.gz' 'https://cmake.org/files/v3.31/cmake-3.31.7-linux-x86_64.tar.gz'
fetch_one ninja-linux 'hex:6f98805688d19672bd699fbbfa2c2cf0fc054ac3df1f0e6a47664d963d530255' 'https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip' 'https://gh-proxy.com/https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip' 'https://ghproxy.net/https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip'

printf 'PREFETCH_COMPLETE count=78\n'
