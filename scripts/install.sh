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
command -v python3 >/dev/null || fail 'python3 is required to read release metadata'
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
# The one place the release repository is named here; the Python checks below get it as an argument.
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
python3 - "$temp/pointer.json" "$target" "$channel" "$repo" "$os" > "$temp/selected" <<'PY'
import json,re,sys
def require(ok):
 if not ok: raise SystemExit('larch setup: invalid channel pointer')
p=json.load(open(sys.argv[1]))
require(p['schema']==1 and p['channel']==sys.argv[3] and isinstance(p['setup'],dict))
# A release that does not build this target has no setup for it in its pointer.
if sys.argv[2] not in p['setup']: raise SystemExit('larch setup: Larch for '+sys.argv[5]+' is coming soon; this release does not include it. See https://github.com/'+sys.argv[4])
a=p['setup'][sys.argv[2]]
require(re.fullmatch('[0-9a-fA-F]{64}',a['sha256']) and isinstance(a['size'],int) and a['size']>0)
url=a['url'];prefix='https://github.com/'+sys.argv[4]+'/releases/download/'
require(url.startswith(prefix))
parts=url[len(prefix):].split('/')
require(len(parts)==2 and all(re.fullmatch('[A-Za-z0-9_.-]+',s) and s not in ('.','..') for s in parts))
print(url);print(a['sha256'].lower());print(a['size']);print(parts[0]);print(parts[1])
PY
selected=()
while IFS= read -r line; do selected+=("$line"); done < "$temp/selected"
url=${selected[0]}
if [ -n "${LARCH_RELEASES_TOKEN:-}" ]; then
  get "$api/releases/tags/${selected[3]}" "$temp/assets.json" application/vnd.github+json
  url=$(python3 - "$temp/assets.json" "${selected[4]}" "$repo" <<'PY'
import json,sys,re
assets=json.load(open(sys.argv[1]))['assets']
a=next(a for a in assets if a['name']==sys.argv[2])
if not re.fullmatch(r'https://api\.github\.com/repos/'+re.escape(sys.argv[3])+r'/releases/assets/[0-9]+',a['url']): raise SystemExit('larch setup: invalid asset URL')
print(a['url'])
PY
)
fi
get "$url" "$temp/larch-setup" application/octet-stream
if command -v sha256sum >/dev/null; then actual=$(sha256sum "$temp/larch-setup"); else actual=$(shasum -a 256 "$temp/larch-setup"); fi
[ "${actual%% *}" = "${selected[1]}" ] || fail 'setup checksum mismatch; retry the install command'
[ "$(wc -c < "$temp/larch-setup" | tr -d ' ')" = "${selected[2]}" ] || fail 'setup size mismatch; retry the install command'
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
