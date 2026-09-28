# 引导包内可移植 Python 3.10.11（Windows embeddable），供「一键环境搭建.bat」调用。
#
# 步骤：下载 embeddable 压缩包（python.org，国内镜像兜底）-> 解压到 python\
#       -> 写入 python310._pth（把 Lib\site-packages 与 ..\app 加入搜索路径）
#       -> 下载 get-pip.py 并安装 pip。
#
# 幂等：python\python.exe 已存在时直接返回。

param(
    [string]$Version = '3.10.11'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PyDir = Join-Path $Root 'python'
$PyExe = Join-Path $PyDir 'python.exe'

if (Test-Path -LiteralPath $PyExe) {
    Write-Host "[bootstrap] 包内 Python 已存在：$PyExe"
    exit 0
}

function Get-File {
    param([string[]]$Urls, [string]$Dest, [long]$MinBytes = 1MB)
    foreach ($url in $Urls) {
        try {
            Write-Host "[bootstrap] 下载 $url"
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $Dest -TimeoutSec 600
            if ((Test-Path -LiteralPath $Dest) -and
                ((Get-Item -LiteralPath $Dest).Length -ge $MinBytes)) {
                return $true
            }
            Write-Host "[bootstrap] 文件不完整：$url"
        } catch {
            Write-Host "[bootstrap] 失败：$($_.Exception.Message)"
        }
    }
    return $false
}

$embName = "python-$Version-embed-amd64.zip"
$embUrls = @(
    "https://www.python.org/ftp/python/$Version/$embName",
    "https://mirrors.huaweicloud.com/python/$Version/$embName"
)

$dlDir = Join-Path $env:TEMP 'ref_forge_bootstrap'
New-Item -ItemType Directory -Force -Path $dlDir | Out-Null
$embZip = Join-Path $dlDir $embName

if (-not (Test-Path -LiteralPath $embZip) -or
    ((Get-Item -LiteralPath $embZip).Length -lt 1MB)) {
    if (-not (Get-File -Urls $embUrls -Dest $embZip -MinBytes 1MB)) {
        Write-Host '[bootstrap] 错误：无法下载 Python 运行时压缩包，请检查网络。'
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path $PyDir | Out-Null
Write-Host "[bootstrap] 解压到 $PyDir"
Expand-Archive -LiteralPath $embZip -DestinationPath $PyDir -Force

$pth = @(
    'python310.zip',
    '.',
    'Lib\site-packages',
    '..\app',
    'import site'
) -join "`r`n"
[System.IO.File]::WriteAllText(
    (Join-Path $PyDir 'python310._pth'), $pth + "`r`n", [System.Text.Encoding]::ASCII)
[System.IO.Directory]::CreateDirectory(
    (Join-Path $PyDir 'Lib\site-packages')) | Out-Null

$getPip = Join-Path $dlDir 'get-pip.py'
if (-not (Test-Path -LiteralPath $getPip) -or
    ((Get-Item -LiteralPath $getPip).Length -lt 10KB)) {
    $pipUrls = @(
        'https://bootstrap.pypa.io/get-pip.py',
        'https://bootstrap.pypa.io/pip/get-pip.py'
    )
    if (-not (Get-File -Urls $pipUrls -Dest $getPip -MinBytes 10KB)) {
        Write-Host '[bootstrap] 错误：无法下载 get-pip.py，请检查网络。'
        exit 1
    }
}

$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
Write-Host '[bootstrap] 安装 pip ...'
& $PyExe $getPip --no-warn-script-location --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    Write-Host '[bootstrap] 错误：pip 安装失败。'
    exit 1
}

& $PyExe -m pip --version
if ($LASTEXITCODE -ne 0) {
    Write-Host '[bootstrap] 错误：pip 不可用。'
    exit 1
}

Remove-Item -LiteralPath $getPip -Force -ErrorAction SilentlyContinue
Write-Host '[bootstrap] 包内 Python 就绪。'
exit 0
