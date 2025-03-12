#!/bin/bash

# Configuration
SERVER="103.49.70.245"
SERVER_USER="root"
GITHUB_REPO="git@github.com:puspendudas/predict.git"
CONTAINER_NAME="api-gateway"
IMAGE_NAME="apigateway"
PORT="8080"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        print_message "✓ $1 successful" "$GREEN"
    else
        print_message "✗ $1 failed" "$RED"
        exit 1
    fi
}

# Ask for GitHub push
read -p "Do you want to push to GitHub first? (y/n): " push_to_github

if [[ $push_to_github == "y" || $push_to_github == "Y" ]]; then
    print_message "Pushing to GitHub..." "$YELLOW"
    
    # Add all changes
    git add .
    check_status "Git add"
    
    # Commit changes
    read -p "Enter commit message: " commit_message
    git commit -m "$commit_message"
    check_status "Git commit"
    
    # Push to GitHub
    git push
    check_status "Git push"
fi

# Deploy to server
print_message "Starting deployment to server..." "$YELLOW"

ssh $SERVER_USER@$SERVER << EOF
    # Stop and remove existing container
    if docker ps -a | grep -q $CONTAINER_NAME; then
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
    
    # Remove existing image
    if docker images | grep -q $IMAGE_NAME; then
        docker rmi $IMAGE_NAME
    fi
    
    # Remove existing code directory
    rm -rf terminal
    
    if [[ "$push_to_github" == "y" || "$push_to_github" == "Y" ]]; then
        # Clone from GitHub
        git clone $GITHUB_REPO
    else
        # Create directory and copy current files
        mkdir -p terminal
        exit
EOF

    if [[ "$push_to_github" != "y" && "$push_to_github" != "Y" ]]; then
        # Use rsync to copy local files to server
        rsync -avz --exclude '.git' --exclude 'node_modules' ./ $SERVER_USER@$SERVER:~/terminal/
    fi

ssh $SERVER_USER@$SERVER << EOF
    cd terminal
    
    # Build and run Docker container
    docker build -t $IMAGE_NAME .
    docker run -d -p $PORT:$PORT --restart unless-stopped --name $CONTAINER_NAME $IMAGE_NAME
    
    # Verify deployment
    if docker ps | grep -q $CONTAINER_NAME; then
        echo "✓ Deployment successful!"
    else
        echo "✗ Deployment failed!"
        exit 1
    fi
EOF

print_message "Deployment process completed!" "$GREEN" 