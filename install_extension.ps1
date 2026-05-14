# install_extension.ps1
# Instala o MemoryMesh como extensao do Claude Desktop automaticamente.
# Uso: powershell -ExecutionPolicy Bypass -File install_extension.ps1

$ErrorActionPreference = "Stop"

$proj        = "C:\Users\carlo\Visual Studio Code\eu\memorymesh"
$ext_dir     = "C:\Users\carlo\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\Claude Extensions"
$ext_name    = "memorymesh"
$exe         = "$proj\.venv\Scripts\memorymesh.exe"
$manifest    = "$proj\memorymesh-extension\manifest.json"
$zip_tmp     = "$proj\memorymesh.zip"
$dxt_file    = "$proj\memorymesh.dxt"
$unpacked    = "$ext_dir\$ext_name"

Write-Host ""
Write-Host "=== MemoryMesh Extension Installer ===" -ForegroundColor Cyan

# 1. Verificar executavel
if (-not (Test-Path $exe)) {
    Write-Error "Executavel nao encontrado: $exe"
    exit 1
}
Write-Host "[OK] Executavel encontrado: $exe"

# 2. Escrever manifest.json correto
Write-Host "[..] Criando manifest.json..."
$manifest_content = @'
{
  "dxt_version": "0.1",
  "name": "memorymesh",
  "display_name": "MemoryMesh",
  "version": "0.1.0",
  "description": "Local MCP hub — semantic search over your personal files",
  "author": {
    "name": "Carlos Coelho",
    "email": ""
  },
  "server": {
    "type": "binary",
    "entry_point": "memorymesh.exe",
    "mcp_config": {
      "command": "C:\\Users\\carlo\\Visual Studio Code\\eu\\memorymesh\\.venv\\Scripts\\memorymesh.exe",
      "args": ["start", "--transport", "stdio"]
    }
  }
}
'@
New-Item -ItemType Directory -Force -Path "$proj\memorymesh-extension" | Out-Null
# Escrever sem BOM — UTF8Encoding($false) é sem BOM, ao contrario de Encoding::UTF8
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($manifest, $manifest_content, $utf8NoBom)
Write-Host "[OK] manifest.json criado (sem BOM)"

# 3. Montar .dxt (zip com manifest + exe stub)
Write-Host "[..] Montando memorymesh.dxt..."
if (Test-Path $zip_tmp) { Remove-Item $zip_tmp -Force }
if (Test-Path $dxt_file) { Remove-Item $dxt_file -Force }

# Adicionar manifest.json ao zip
Compress-Archive -Path $manifest -DestinationPath $zip_tmp

# Renomear para .dxt
Rename-Item $zip_tmp $dxt_file
Write-Host "[OK] memorymesh.dxt criado em $dxt_file"

# 4. Instalar como extensao descompactada (copia pasta direto)
Write-Host "[..] Instalando extensao descompactada em Claude Extensions..."
if (Test-Path $unpacked) {
    Remove-Item $unpacked -Recurse -Force
    Write-Host "    (removida versao anterior)"
}
New-Item -ItemType Directory -Force -Path $unpacked | Out-Null
Copy-Item $manifest "$unpacked\manifest.json" -Force
Write-Host "[OK] Extensao copiada para: $unpacked"

# 5. Resultado
Write-Host ""
Write-Host "=== Instalacao concluida! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Proximos passos:" -ForegroundColor Yellow
Write-Host "  1. Feche o Claude Desktop pelo system tray (botao direito -> Quit)"
Write-Host "  2. Reabra o Claude Desktop"
Write-Host "  3. Va em Configuracoes -> Extensoes e verifique se 'MemoryMesh' aparece"
Write-Host ""
Write-Host "Se a extensao descompactada nao aparecer, use o arquivo .dxt manualmente:"
Write-Host "  $dxt_file"
