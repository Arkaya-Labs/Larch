#!/usr/bin/env bash
# The downloaded setup authenticates the pointer signature and its own bytes before installing.
set +x
set -euo pipefail
fail() { echo "larch setup: $1" >&2; exit 1; }
case "$(uname -s):$(uname -m)" in
  Linux:x86_64|Linux:amd64)
    [ "$(getconf LONG_BIT)" = 64 ] || fail '32-bit Linux is unsupported'
    target=x86_64-unknown-linux-gnu os=Linux ;;
  Darwin:arm64|Darwin:aarch64) target=aarch64-apple-darwin os=macOS ;;
  Darwin:*) fail 'Intel Mac is unsupported; Apple silicon is required' ;;
  *) fail 'unsupported OS or CPU; use 64-bit x86 Linux or Apple silicon macOS' ;;
esac
command -v curl >/dev/null || fail 'curl is required'
# Prints the scalar at a dotted key path (list items by index) of a JSON file, or with a third
# argument `json` any value as JSON; fails when the key is absent. macOS reads JSON with its own
# plutil, because its python3 is a stub that, on a Mac without developer tools, stops the install
# to download them first.
if [ "$os" = macOS ]; then
  json() { plutil -extract "$2" "${3:-raw}" -o - "$1" 2>/dev/null; }
else
  command -v python3 >/dev/null || fail 'python3 is required to read release metadata'
  json() {
    python3 - "$1" "$2" "${3:-raw}" 2>/dev/null <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'): v=v[int(k)] if isinstance(v,list) else v[k]
if sys.argv[3]=='json': print(json.dumps(v)); raise SystemExit
if isinstance(v,(dict,list)): raise SystemExit(1)
print(str(v).lower() if isinstance(v,bool) else v)
PY
  }
fi
channel=stable
next=0
# With no options but --channel, setup opens its window; --terminal, any other option, or no
# display keeps it here. --terminal is this script's own and is not passed on.
terminal=0
pass=()
for arg in "$@"; do
  if [ "$next" = 1 ]; then channel=$arg; next=0; pass+=("$arg"); continue; fi
  case "$arg" in
    --channel) next=1 ;;
    --channel=*) channel=${arg#--channel=} ;;
    --terminal) terminal=1; continue ;;
    *) terminal=1 ;;
  esac
  pass+=("$arg")
done
case "$channel" in ''|*[!a-zA-Z0-9_-]*) fail 'invalid channel' ;; esac
[ "$next" = 0 ] || fail '--channel requires a value'
temp=$(mktemp -d)
trap 'rm -r -- "$temp"' EXIT
# The one place the release repository is named here.
repo=Arkaya-Labs/Larch
api=https://api.github.com/repos/$repo
get() {
  local url=$1 dest=$2 accept=$3
  if [ -n "${LARCH_RELEASES_TOKEN:-}" ]; then
    case "$LARCH_RELEASES_TOKEN" in *[!a-zA-Z0-9_.-]*) fail 'invalid release token' ;; esac
    # Config arrives on stdin, so the credential is never a file or a command argument.
    local redirect
    redirect=$(printf 'header = "Authorization: Bearer %s"\n' "$LARCH_RELEASES_TOKEN" |
      curl --config - --fail --silent --show-error --proto '=https' \
        --user-agent larch-setup-bootstrap --header "Accept: $accept" --output "$dest" \
        --write-out '%{redirect_url}' "$url")
    if [ -n "$redirect" ]; then
      case "$redirect" in https://*) ;; *) fail 'release redirect must use HTTPS' ;; esac
      curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
        --user-agent larch-setup-bootstrap --header "Accept: $accept" --output "$dest" "$redirect"
    fi
  else
    curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
      --user-agent larch-setup-bootstrap --header "Accept: $accept" --output "$dest" "$url"
  fi
}
if [ -n "${LARCH_RELEASES_TOKEN:-}" ]; then
  get "$api/contents/$channel.json?ref=channels" "$temp/pointer.json" application/vnd.github.raw+json
else
  get "https://raw.githubusercontent.com/$repo/channels/$channel.json" "$temp/pointer.json" application/json
fi
bad() { fail 'invalid channel pointer'; }
pointer=$temp/pointer.json
[ "$(json "$pointer" schema)" = 1 ] && [ "$(json "$pointer" channel)" = "$channel" ] || bad
# A release that does not build this target has no setup for it in its pointer.
json "$pointer" "setup.$target" json >/dev/null ||
  fail "Larch for $os is coming soon; this release does not include it. See https://github.com/$repo"
url=$(json "$pointer" "setup.$target.url") && sha=$(json "$pointer" "setup.$target.sha256") &&
  size=$(json "$pointer" "setup.$target.size") || bad
[[ $sha =~ ^[0-9a-fA-F]{64}$ && $size =~ ^[1-9][0-9]*$ ]] || bad
prefix=https://github.com/$repo/releases/download/
[[ $url == "$prefix"* ]] || bad
tag=${url#"$prefix"} name=${tag#*/} tag=${tag%%/*}
for part in "$tag" "$name"; do
  [[ $part =~ ^[A-Za-z0-9_.-]+$ && $part != . && $part != .. ]] || bad
done
sha=$(printf %s "$sha" | tr A-F a-f)
if [ -n "${LARCH_RELEASES_TOKEN:-}" ]; then
  get "$api/releases/tags/$tag" "$temp/assets.json" application/vnd.github+json
  i=0
  while asset=$(json "$temp/assets.json" "assets.$i.name"); do
    [ "$asset" = "$name" ] && break
    i=$((i + 1))
  done
  [ "$asset" = "$name" ] || fail 'release has no setup asset'
  url=$(json "$temp/assets.json" "assets.$i.url") || fail 'invalid asset URL'
  [[ $url =~ ^https://api\.github\.com/repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/releases/assets/[0-9]+$ &&
     $url == "$api/releases/assets/"* ]] || fail 'invalid asset URL'
fi
get "$url" "$temp/larch-setup" application/octet-stream
if command -v sha256sum >/dev/null; then actual=$(sha256sum "$temp/larch-setup"); else actual=$(shasum -a 256 "$temp/larch-setup"); fi
[ "${actual%% *}" = "$sha" ] || fail 'setup checksum mismatch; retry the install command'
[ "$(wc -c < "$temp/larch-setup" | tr -d ' ')" = "$size" ] || fail 'setup size mismatch; retry the install command'
chmod 700 "$temp/larch-setup"
# A window needs a desktop: a Mac's own session, or Linux with X11 or Wayland.
case "$target" in
  *apple*) if [ -n "${SSH_CONNECTION:-}${SSH_TTY:-}" ]; then terminal=1; fi ;;
  *) if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then terminal=1; fi ;;
esac
if [ "$terminal" = 0 ]; then
  "$temp/larch-setup" window --bootstrap ${pass[@]+"${pass[@]}"} && exit 0
  echo 'larch setup: the installer window did not open, so setup continues here' >&2
fi
if ( : </dev/tty ) 2>/dev/null; then
  "$temp/larch-setup" install --bootstrap ${pass[@]+"${pass[@]}"} </dev/tty
else
  "$temp/larch-setup" install --bootstrap ${pass[@]+"${pass[@]}"}
fi
