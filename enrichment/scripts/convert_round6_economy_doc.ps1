param(
    [Parameter(Mandatory=$true)][string]$InputDirectory,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$inputRoot = (Resolve-Path -LiteralPath $InputDirectory).Path
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $outputRoot) { throw 'Choose a new output directory' }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$wordApp = $null
$records = @()
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordApp.AutomationSecurity = 3
    $wordApp.Options.UpdateLinksAtOpen = $false
    $wordApp.Options.SaveNormalPrompt = $false
    foreach ($inputFile in Get-ChildItem -LiteralPath $inputRoot -Filter '*.doc' | Sort-Object Name) {
        $document = $null
        try {
            $beforeHash = (Get-FileHash -LiteralPath $inputFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            $document = $wordApp.Documents.Open($inputFile.FullName, $false, $true, $false)
            $docxPath = Join-Path $outputRoot ($inputFile.BaseName + '.docx')
            $pdfPath = Join-Path $outputRoot ($inputFile.BaseName + '.pdf')
            $document.SaveAs2($docxPath, 16)
            $document.ExportAsFixedFormat($pdfPath, 17)
        } finally {
            if ($null -ne $document) { $document.Close(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
        }
        $records += @{
            input = $inputFile.Name
            input_sha256 = $beforeHash
            docx = [IO.Path]::GetFileName($docxPath)
            docx_sha256 = (Get-FileHash -LiteralPath $docxPath -Algorithm SHA256).Hash.ToLowerInvariant()
            pdf = [IO.Path]::GetFileName($pdfPath)
            pdf_sha256 = (Get-FileHash -LiteralPath $pdfPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        if ((Get-FileHash -LiteralPath $inputFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -ne $beforeHash) { throw 'Original changed' }
    }
    @{
        tool = 'Microsoft Word COM read-only conversion'
        version = $wordApp.Version
        build = $wordApp.Build
        automation_security = 3
        update_links_at_open = $false
        original_documents_unchanged = $true
        converted_at_utc = [DateTime]::UtcNow.ToString('o')
        records = $records
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputRoot 'conversion_manifest.json') -Encoding utf8
    Write-Output ('Converted documents: ' + $records.Count)
} finally {
    if ($null -ne $wordApp) { $wordApp.Quit(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) }
}
