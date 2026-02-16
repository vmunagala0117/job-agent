$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile('scripts\deploy.ps1',[ref]$null,[ref]$errors)
if ($errors -ne $null -and $errors.Count -gt 0) {
    $errors | Format-List
    exit 1
} else {
    Write-Output 'No parse errors'
}
