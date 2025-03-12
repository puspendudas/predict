# Configuration
$SERVER = "103.49.70.245"
$SERVER_USER = "root"
$GITHUB_REPO = "git@github.com:puspendudas/predict.git"
$CONTAINER_NAME = "api-gateway"
$IMAGE_NAME = "apigateway"
$PORT = "8080"

# Function to print colored messages
function Write-ColoredMessage {
    param(
        [string]$Message,
        [string]$Color
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to check command status
function Check-Status {
    param(
        [string]$Operation
    )
    if ($LASTEXITCODE -eq 0) {
        Write-ColoredMessage "✓ $Operation successful" "Green"
    }
    else {
        Write-ColoredMessage "✗ $Operation failed" "Red"
        exit 1
    }
}  # ← Fixed missing closing brace

# Ask for GitHub push
$push_to_github = Read-Host "Do you want to push to GitHub first? (y/n)"

if ($push_to_github -eq "y" -or $push_to_github -eq "Y") {
    Write-ColoredMessage "Pushing to GitHub..." "Yellow"
    
    # Add all changes
    git add .
    Check-Status "Git add"
    
    # Commit changes
    $commit_message = Read-Host "Enter commit message"
    git commit -m $commit_message
    Check-Status "Git commit"
    
    # Push to GitHub
    git push
    Check-Status "Git push"
}

# Deploy to server
Write-ColoredMessage "Starting deployment to server..." "Yellow"

# Create SSH commands file
$sshCommands = @"
#!/bin/bash
set -e  # Exit on error

# Stop and remove existing container if running
if [[ \$(docker ps -a --format '{{.Names}}' | grep -w "${CONTAINER_NAME}") ]]; then
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

# Remove existing image if exists
if [[ \$(docker images --format '{{.Repository}}' | grep -w "${IMAGE_NAME}") ]]; then
    docker rmi ${IMAGE_NAME}
fi

# Remove existing code directory
rm -rf predict

# Clone from GitHub if push was done, else use direct copy
if [[ "$push_to_github" = "y" || "$push_to_github" = "Y" ]]; then
    git clone ${GITHUB_REPO} predict
else
    mkdir -p predict
fi

cd predict

# Build and run Docker container
docker build -t ${IMAGE_NAME} .
docker run -d -p ${PORT}:${PORT} --restart unless-stopped --name ${CONTAINER_NAME} ${IMAGE_NAME}

# Verify deployment
if docker ps --format '{{.Names}}' | grep -w "${CONTAINER_NAME}"; then
    echo "✓ Deployment successful!"
else
    echo "✗ Deployment failed!"
    exit 1
fi
"@

# Save SSH commands to a temporary file
$sshCommandsFile = [System.IO.Path]::GetTempFileName()
$sshCommands | Out-File -FilePath $sshCommandsFile -Encoding ASCII

if ($push_to_github -ne "y" -and $push_to_github -ne "Y") {
    # For direct file transfer, use rsync instead of scp
    Write-ColoredMessage "Copying files directly to server..." "Yellow"
    & rsync -av --exclude '.git' --exclude 'node_modules' ./ "${SERVER_USER}@${SERVER}:~/predict/"
}

# Execute SSH commands correctly in PowerShell
Get-Content $sshCommandsFile | ssh ${SERVER_USER}@${SERVER} "bash -s"

# Clean up temporary file
Remove-Item $sshCommandsFile -Force

Write-ColoredMessage "Deployment process completed!" "Green"