<#
.SYNOPSIS
    Deploys the Job Agent Web App (container) and Azure Function to Azure.

.DESCRIPTION
    This script handles the full deployment pipeline:
      1. Creates an Azure Container Registry (ACR) if it doesn't exist
      2. Builds and pushes the Docker image to ACR
      3. Creates an App Service Plan + Web App (container) if they don't exist
      4. Configures all Web App settings (OpenAI, DB, telemetry, cron key)
      5. Deploys the Azure Function (zip deploy) if requested
      6. Configures Function App settings (CRON_APP_URL, CRON_API_KEY)

.PARAMETER ResourceGroup
    Azure resource group name (default: vm-demo-agent-rg)

.PARAMETER Location
    Azure region (default: eastus2)

.PARAMETER WebAppName
    Name for the Web App (default: job-agent-webapp)

.PARAMETER FunctionAppName
    Name of the existing Function App (default: jobdemo01)

.PARAMETER AcrName
    Azure Container Registry name - must be globally unique, lowercase alphanumeric
    (default: jobagentacr01)

.PARAMETER AppServicePlanName
    App Service Plan name for the Web App (default: job-agent-plan)

.PARAMETER AppServiceSku
    App Service Plan SKU (default: B1 - Basic, good for demos)

.PARAMETER ImageTag
    Docker image tag (default: latest)

.PARAMETER DeployFunction
    Also deploy/redeploy the Azure Function (default: $false)

.PARAMETER SkipWebApp
    Skip Web App deployment, only configure settings or deploy function (default: $false)

.PARAMETER CronApiKey
    Shared secret for cron authentication. If not provided, generates a random one.

.PARAMETER EnvFile
    Path to .env file for reading secrets (default: .env in repo root)

.EXAMPLE
    # Full first-time deployment
    .\scripts\deploy.ps1

    # Redeploy just the webapp (code update)
    .\scripts\deploy.ps1

    # Redeploy both webapp and function
    .\scripts\deploy.ps1 -DeployFunction

    # Deploy function only
    .\scripts\deploy.ps1 -SkipWebApp -DeployFunction
#>

[CmdletBinding()]
param(
    [string]$ResourceGroup     = "vm-demo-agent-rg",
    [string]$Location          = "eastus2",
    [string]$WebAppName        = "job-agent-webapp",
    [string]$FunctionAppName   = "jobdemo01",
    [string]$AcrName           = "jobagentacr01",
    [string]$AppServicePlanName = "job-agent-plan",
    [string]$AppServiceSku     = "B1",
    [string]$ImageTag          = "latest",
    [switch]$DeployFunction,
    [switch]$SkipWebApp,
    [string]$CronApiKey        = "",
    [string]$EnvFile           = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

function Write-Step   { param([string]$msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok     { param([string]$msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn   { param([string]$msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Err    { param([string]$msg) Write-Host "  [ERROR] $msg" -ForegroundColor Red }

function Test-AzResource {
    <# Returns $true if an az command produces output, $false otherwise #>
    param([string]$Cmd)

    # Use a child PowerShell process and capture output/errors safely
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell.exe"
    $psi.Arguments = "-NoProfile -Command $Cmd"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.Start() | Out-Null
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()

    if ($proc.ExitCode -ne 0) { return $false }
    $result = $stdout | Out-String
    return ($result.Trim().Length -gt 0 -and $result.Trim() -ne "[]")
}

function Read-EnvFile {
    <# Parse a .env file into a hashtable #>
    param([string]$Path)
    $vars = @{}
    if (-not (Test-Path $Path)) { return $vars }
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $eqIdx = $line.IndexOf("=")
            if ($eqIdx -gt 0) {
                $key = $line.Substring(0, $eqIdx).Trim()
                $val = $line.Substring($eqIdx + 1).Trim()
                $vars[$key] = $val
            }
        }
    }
    return $vars
}

# ──────────────────────────────────────────────────────────────────
# Resolve paths and load config
# ──────────────────────────────────────────────────────────────────

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dockerfilePath = Join-Path $repoRoot "Dockerfile"
$funcDir = Join-Path $repoRoot "azure-functions\daily-search"

if (-not $EnvFile) { $EnvFile = Join-Path $repoRoot ".env" }

Write-Host ""
Write-Host "=== JOB AGENT - AZURE DEPLOYMENT ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Resource Group:   $ResourceGroup"
Write-Host "  Location:         $Location"
Write-Host "  Web App:          $WebAppName"
Write-Host "  Function App:     $FunctionAppName"
Write-Host "  ACR:              $AcrName"
Write-Host "  Deploy Function:  $DeployFunction"
Write-Host "  Skip Web App:     $SkipWebApp"
Write-Host ""

# Load environment variables from .env
$envVars = Read-EnvFile $EnvFile
if ($envVars.Count -gt 0) {
    Write-Ok "Loaded $($envVars.Count) variables from $EnvFile"
} else {
    Write-Warn "No .env file found at $EnvFile - using defaults/prompts"
}

# Generate or use provided CRON_API_KEY
if (-not $CronApiKey) {
    if ($envVars.ContainsKey("CRON_API_KEY") -and $envVars["CRON_API_KEY"] -ne "test-local-key") {
        $CronApiKey = $envVars["CRON_API_KEY"]
        Write-Ok "Using CRON_API_KEY from .env"
    } else {
        $CronApiKey = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
        Write-Ok "Generated new CRON_API_KEY: $($CronApiKey.Substring(0,8))..."
    }
}

# Validate Azure CLI login
Write-Step "Checking Azure CLI login"
try {
    $account = az account show --query "{name:name, id:id}" -o json 2>$null | ConvertFrom-Json
    Write-Ok "Logged in as: $($account.name) ($($account.id))"
} catch {
    Write-Err "Not logged in to Azure CLI. Run: az login"
    exit 1
}

# ──────────────────────────────────────────────────────────────────
# Step 1: Azure Container Registry
# ──────────────────────────────────────────────────────────────────

if (-not $SkipWebApp) {
    Write-Step "Step 1: Azure Container Registry ($AcrName)"

    $acrExists = Test-AzResource "az acr show --name $AcrName --resource-group $ResourceGroup --query name -o tsv"

    if ($acrExists) {
        Write-Ok "ACR '$AcrName' already exists"
    } else {
        Write-Host "  Creating ACR '$AcrName'..."
        az acr create `
            --resource-group $ResourceGroup `
            --name $AcrName `
            --sku Basic `
            --location $Location `
            --admin-enabled true `
            --output none
        Write-Ok "ACR '$AcrName' created"
    }

    $acrServer = az acr show --name $AcrName --query loginServer -o tsv
    $imageFull = "${acrServer}/job-agent:${ImageTag}"
    Write-Ok "Image: $imageFull"

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Build and push Docker image
    # ──────────────────────────────────────────────────────────────────

    Write-Step "Step 2: Build and push Docker image"
    Write-Host "  Building image via ACR Tasks (cloud build)..."
    Write-Host "  This may take 2-5 minutes on first build..."

    az acr build `
        --registry $AcrName `
        --image "job-agent:${ImageTag}" `
        --file $dockerfilePath `
        $repoRoot

    Write-Ok "Image pushed: $imageFull"

    # ──────────────────────────────────────────────────────────────────
    # Step 3: App Service Plan
    # ──────────────────────────────────────────────────────────────────

    Write-Step "Step 3: App Service Plan ($AppServicePlanName)"

    $planExists = Test-AzResource "az appservice plan show --name $AppServicePlanName --resource-group $ResourceGroup --query name -o tsv"

    if ($planExists) {
        Write-Ok "Plan '$AppServicePlanName' already exists"
    } else {
        Write-Host "  Creating App Service Plan ($AppServiceSku - Linux)..."
        az appservice plan create `
            --name $AppServicePlanName `
            --resource-group $ResourceGroup `
            --location $Location `
            --sku $AppServiceSku `
            --is-linux `
            --output none
        Write-Ok "Plan '$AppServicePlanName' created"
    }

    # ──────────────────────────────────────────────────────────────────
    # Step 4: Web App (Container)
    # ──────────────────────────────────────────────────────────────────

    Write-Step "Step 4: Web App ($WebAppName)"

    $webappExists = Test-AzResource "az webapp show --name $WebAppName --resource-group $ResourceGroup --query name -o tsv"

    if ($webappExists) {
        Write-Ok "Web App '$WebAppName' already exists - updating container image"
        az webapp config container set `
            --name $WebAppName `
            --resource-group $ResourceGroup `
            --container-image-name $imageFull `
            --container-registry-url "https://$acrServer" `
            --output none
    } else {
        Write-Host "  Creating Web App with container image..."

        # Get ACR credentials for the Web App
        $acrUser = az acr credential show --name $AcrName --query username -o tsv
        $acrPass = az acr credential show --name $AcrName --query "passwords[0].value" -o tsv

        az webapp create `
            --name $WebAppName `
            --resource-group $ResourceGroup `
            --plan $AppServicePlanName `
            --container-image-name $imageFull `
            --container-registry-url "https://$acrServer" `
            --container-registry-user $acrUser `
            --container-registry-password $acrPass `
            --output none

        # Set the port (Azure Web App defaults to 80, our app uses 8080)
        az webapp config appsettings set `
            --name $WebAppName `
            --resource-group $ResourceGroup `
            --settings WEBSITES_PORT=8080 `
            --output none

        Write-Ok "Web App '$WebAppName' created"
    }

    # ──────────────────────────────────────────────────────────────────
    # Step 5: Configure Web App settings
    # ──────────────────────────────────────────────────────────────────

    Write-Step "Step 5: Configure Web App settings"

    # Helper to read .env values with a default (compatible with older PowerShell)
    function Get-EnvValue { param([string]$k, [string]$d) if ($envVars.ContainsKey($k) -and $envVars[$k]) { return $envVars[$k] } else { return $d } }

    $AZURE_OPENAI_ENDPOINT = Get-EnvValue 'AZURE_OPENAI_ENDPOINT' 'https://vm-demo-fdry-01.openai.azure.com'
    $AZURE_OPENAI_DEPLOYMENT_NAME = Get-EnvValue 'AZURE_OPENAI_DEPLOYMENT_NAME' 'gpt-5.2'
    $AZURE_OPENAI_API_KEY = Get-EnvValue 'AZURE_OPENAI_API_KEY' ''
    $AZURE_OPENAI_API_VERSION = Get-EnvValue 'AZURE_OPENAI_API_VERSION' '2024-05-01-preview'
    $AZURE_OPENAI_EMBEDDING_MODEL = Get-EnvValue 'AZURE_OPENAI_EMBEDDING_MODEL' 'text-embedding-3-small'
    $SERPAPI_API_KEY = Get-EnvValue 'SERPAPI_API_KEY' ''
    $DB_HOST = Get-EnvValue 'DB_HOST' 'pgvector-demo-01-db.postgres.database.azure.com'
    $DB_PORT = Get-EnvValue 'DB_PORT' '5432'
    $DB_NAME = Get-EnvValue 'DB_NAME' 'job_agent'
    $DB_USER = Get-EnvValue 'DB_USER' 'pgadmin'
    $DB_PASSWORD = Get-EnvValue 'DB_PASSWORD' ''
    $DB_SSL_MODE = Get-EnvValue 'DB_SSL_MODE' 'require'
    $APPLICATIONINSIGHTS_CONNECTION_STRING = Get-EnvValue 'APPLICATIONINSIGHTS_CONNECTION_STRING' ''

    $webAppSettings = @(
        "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME=$AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDING_MODEL=$AZURE_OPENAI_EMBEDDING_MODEL",
        "SERPAPI_API_KEY=$SERPAPI_API_KEY",
        "DB_HOST=$DB_HOST",
        "DB_PORT=$DB_PORT",
        "DB_NAME=$DB_NAME",
        "DB_USER=$DB_USER",
        "DB_PASSWORD=$DB_PASSWORD",
        "DB_SSL_MODE=$DB_SSL_MODE",
        "APPLICATIONINSIGHTS_CONNECTION_STRING=$APPLICATIONINSIGHTS_CONNECTION_STRING",
        "ENABLE_INSTRUMENTATION=true",
        "ENABLE_CONSOLE_EXPORTERS=false",
        "ENABLE_SENSITIVE_DATA=false",
        "OTEL_SERVICE_NAME=job-agent",
        "CRON_API_KEY=$CronApiKey",
        "WEBSITES_PORT=8080"
    )

    az webapp config appsettings set `
        --name $WebAppName `
        --resource-group $ResourceGroup `
        --settings @webAppSettings `
        --output none

    Write-Ok "Web App settings configured ($($webAppSettings.Count) settings)"

    # Enable always-on so the webapp is warm when the cron timer fires
    az webapp config set `
        --name $WebAppName `
        --resource-group $ResourceGroup `
        --always-on true `
        --output none 2>$null

    Write-Ok "Always-on enabled"

    $webAppUrl = "https://${WebAppName}.azurewebsites.net"
    Write-Ok "Web App URL: $webAppUrl"
}

# ──────────────────────────────────────────────────────────────────
# Step 6: Deploy Azure Function (if requested)
# ──────────────────────────────────────────────────────────────────

if ($DeployFunction) {
    Write-Step "Step 6: Deploy Azure Function ($FunctionAppName)"

    if (-not (Test-Path $funcDir)) {
        Write-Err "Function directory not found: $funcDir"
        exit 1
    }

    # Verify the Function App exists
    $funcExists = Test-AzResource "az functionapp show --name $FunctionAppName --resource-group $ResourceGroup --query name -o tsv"
    if (-not $funcExists) {
        Write-Err "Function App '$FunctionAppName' not found in resource group '$ResourceGroup'."
        Write-Err "Create it first in the Azure Portal or with 'az functionapp create'."
        exit 1
    }

    # Create zip for deployment
    $zipPath = Join-Path $repoRoot "azure-functions\daily-search-deploy.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

    Write-Host "  Creating deployment zip..."
    $filesToZip = Get-ChildItem -Path $funcDir -File |
        Where-Object { $_.Name -notin @(".env", ".gitignore") -and $_.Extension -ne ".pyc" } |
        Where-Object { $_.Name -ne "daily-search-deploy.zip" }

    Compress-Archive -Path ($filesToZip.FullName) -DestinationPath $zipPath -Force
    $zipSize = [math]::Round((Get-Item $zipPath).Length / 1KB, 1)
    Write-Ok "Deployment zip: $zipPath ($zipSize KB)"

    Write-Host "  Deploying to '$FunctionAppName'..."
    az functionapp deployment source config-zip `
        --resource-group $ResourceGroup `
        --name $FunctionAppName `
        --src $zipPath `
        --output none

    Write-Ok "Function deployed"

    # Clean up zip
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
}

# ──────────────────────────────────────────────────────────────────
# Step 7: Configure Function App settings
# ──────────────────────────────────────────────────────────────────

if ($DeployFunction -or -not $SkipWebApp) {
    Write-Step "Step 7: Configure Function App settings"

    # Determine the Web App URL
    if (-not $SkipWebApp) {
        $cronAppUrl = "https://${WebAppName}.azurewebsites.net"
    } else {
        # Try to read existing CRON_APP_URL from function settings
        $existingUrl = az functionapp config appsettings list `
            --name $FunctionAppName `
            --resource-group $ResourceGroup `
            --query "[?name=='CRON_APP_URL'].value | [0]" -o tsv 2>$null
        $cronAppUrl = if ($existingUrl) { $existingUrl } else { "https://${WebAppName}.azurewebsites.net" }
    }

    $funcSettings = @(
        "CRON_APP_URL=$cronAppUrl"
        "CRON_API_KEY=$CronApiKey"
        "CRON_SCHEDULE=0 0 6 * * *"
    )

    az functionapp config appsettings set `
        --name $FunctionAppName `
        --resource-group $ResourceGroup `
        --settings @funcSettings `
        --output none

    Write-Ok "Function settings configured"
    Write-Ok "CRON_APP_URL = $cronAppUrl"
    Write-Ok "CRON_SCHEDULE = 0 0 6 * * * (daily at 6:00 AM UTC)"
}

# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== DEPLOYMENT COMPLETE ===" -ForegroundColor Green
Write-Host ""

if (-not $SkipWebApp) {
    $webAppUrl = "https://${WebAppName}.azurewebsites.net"
    Write-Host "  Web App:       $webAppUrl" -ForegroundColor White
    Write-Host "  Chat UI:       $webAppUrl/" -ForegroundColor White
    Write-Host "  Health check:  $webAppUrl/health" -ForegroundColor White
    Write-Host "  API docs:      $webAppUrl/docs" -ForegroundColor White
}

if ($DeployFunction) {
    Write-Host "  Function App:  $FunctionAppName (timer: daily 6:00 AM UTC)" -ForegroundColor White
}

Write-Host ""
Write-Host "  CRON_API_KEY:  $($CronApiKey.Substring(0, [Math]::Min(8, $CronApiKey.Length)))..." -ForegroundColor DarkGray
Write-Host ""

# Post-deployment verification
if (-not $SkipWebApp) {
    Write-Host "  Next steps:" -ForegroundColor Yellow
    Write-Host "    1. Verify health: Invoke-RestMethod $webAppUrl/health" -ForegroundColor Yellow
    Write-Host "    2. Check logs:    az webapp log tail --name $WebAppName --resource-group $ResourceGroup" -ForegroundColor Yellow
    $testCronCmd = "Invoke-RestMethod -Uri '$webAppUrl/api/cron/daily-search' -Method Post -Headers @{ 'X-Cron-Key' = '$($CronApiKey.Substring(0,8))...' }"
    Write-Host "    3. Test cron:     $testCronCmd" -ForegroundColor Yellow
    Write-Host ""
}
