Write-Host 'Planned Azure / file operations from scripts/deploy.ps1:'
$patterns = @(
    'az acr create',
    'az acr build',
    'az webapp create',
    'az webapp config appsettings set',
    'az webapp config container set',
    'az appservice plan create',
    'az functionapp deployment source config-zip',
    'az functionapp config appsettings set',
    'Compress-Archive'
)
Get-Content 'scripts\deploy.ps1' | Select-String -Pattern ($patterns -join '|') -AllMatches | ForEach-Object { $_.Line.Trim() } 
