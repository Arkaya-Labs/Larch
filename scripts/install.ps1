# Setup verifies the primary signature and its own hash after this HTTPS bootstrap.
$ErrorActionPreference = 'Stop'
$VerbosePreference = 'SilentlyContinue'
$DebugPreference = 'SilentlyContinue'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or
    -not [Environment]::Is64BitProcess -or
    [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() -ne 'X64') {
    [Console]::Error.WriteLine('larch setup: unsupported OS or CPU; 64-bit x86 Windows is required'); exit 1
}
# The one place the release repository is named here.
$repo = 'Arkaya-Labs/Larch'
$repoPattern = [regex]::Escape($repo)
$target = 'x86_64-pc-windows-msvc'
$channel = 'stable'
$comingSoon = $false
# With no options but --channel, setup opens its window; --terminal or any other option keeps it
# here. --terminal is this script's own and is not passed on.
$terminal = $false
$pass = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq '--channel') { $i++; if ($i -ge $args.Count) { throw '--channel requires a value' }; $channel = $args[$i]; $pass += '--channel', $channel }
    elseif ($args[$i] -like '--channel=*') { $channel = $args[$i].Substring(10); $pass += $args[$i] }
    elseif ($args[$i] -eq '--terminal') { $terminal = $true }
    else { $terminal = $true; $pass += $args[$i] }
}
if ($channel -notmatch '^[A-Za-z0-9_-]+$') { throw 'invalid channel' }
$token = $env:LARCH_RELEASES_TOKEN
if ($token -and $token -match '\s') { throw 'invalid release token' }
$temp = Join-Path ([IO.Path]::GetTempPath()) ('larch-setup-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp | Out-Null
function Get-ReleaseFile([string]$Url, [string]$File, [string]$Accept) {
    $headers = @{ Accept = $Accept; 'User-Agent' = 'larch-setup-bootstrap' }
    if ($token) { $headers.Authorization = 'Bearer ' + $token }
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -Headers $headers -MaximumRedirection 0 -OutFile $File | Out-Null
    } catch {
        $response = $_.Exception.Response
        if ($null -eq $response -or [int]$response.StatusCode -lt 300 -or [int]$response.StatusCode -ge 400) { throw 'release download failed; check the connection and retry' }
        $location = [Uri]::new([Uri]$Url, [string]$response.Headers['Location'])
        if ($location.Scheme -ne 'https') { throw 'release redirect must use HTTPS' }
        # An authenticated request never forwards its credential, including same-host redirects.
        $headers.Remove('Authorization')
        Invoke-WebRequest -UseBasicParsing -Uri $location -Headers $headers -MaximumRedirection 5 -OutFile $File | Out-Null
    }
}
try {
    $api = "https://api.github.com/repos/$repo"
    $pointerFile = Join-Path $temp pointer.json
    if ($token) { Get-ReleaseFile "$api/contents/$channel.json`?ref=channels" $pointerFile 'application/vnd.github.raw+json' }
    else { Get-ReleaseFile "https://raw.githubusercontent.com/$repo/channels/$channel.json" $pointerFile 'application/json' }
    $pointer = Get-Content -Raw -LiteralPath $pointerFile | ConvertFrom-Json
    if ($pointer.schema -ne 1 -or $pointer.channel -cne $channel) { throw 'unsupported pointer or wrong channel' }
    $setup = $pointer.setup.$target
    # A release that does not build Windows has no Windows setup in its pointer.
    if ($null -ne $pointer.setup -and $null -eq $setup) { $comingSoon = $true; throw 'no Windows build' }
    if ($null -eq $setup -or $setup.sha256 -notmatch '^[0-9a-fA-F]{64}$' -or $setup.size -le 0) { throw 'channel has no valid setup for this target' }
    $url = [string]$setup.url
    if ($url -notmatch ('^https://github\.com/' + $repoPattern + '/releases/download/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$')) { throw 'setup URL is outside the release repository' }
    $tag = $Matches[1]; $name = $Matches[2]
    if ($tag -in @('.','..') -or $name -in @('.','..')) { throw 'invalid release path' }
    if ($token) {
        $assetsFile = Join-Path $temp assets.json
        Get-ReleaseFile "$api/releases/tags/$tag" $assetsFile 'application/vnd.github+json'
        $release = Get-Content -Raw -LiteralPath $assetsFile | ConvertFrom-Json
        $asset = @($release.assets | Where-Object { $_.name -ceq $name })
        if ($asset.Count -ne 1 -or $asset[0].url -notmatch ('^https://api\.github\.com/repos/' + $repoPattern + '/releases/assets/[0-9]+$')) { throw 'setup asset is missing' }
        $url = $asset[0].url
    }
    $exe = Join-Path $temp larch-setup.exe
    Get-ReleaseFile $url $exe 'application/octet-stream'
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash -ine $setup.sha256 -or (Get-Item -LiteralPath $exe).Length -ne $setup.size) { throw 'setup checksum or size mismatch; retry the install command' }
    if (-not $terminal) {
        & $exe window --bootstrap @pass
        if ($LASTEXITCODE -ne 0) { [Console]::Error.WriteLine('larch setup: the installer window did not open, so setup continues here'); $terminal = $true }
    }
    if ($terminal) { & $exe install --bootstrap @pass }
    $result = $LASTEXITCODE
} catch {
    if ($comingSoon) { [Console]::Error.WriteLine("larch setup: Larch for Windows is coming soon; this release does not include it. See https://github.com/$repo") } else { [Console]::Error.WriteLine('larch setup: download or verification failed; check the release source and retry') }
    $result = 1
} finally {
    Remove-Item -LiteralPath $temp -Recurse -Force
}
exit $result
