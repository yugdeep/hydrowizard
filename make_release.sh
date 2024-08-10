#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [patch|minor|major]"
    echo "  patch: Increment the patch version (0.0.X)"
    echo "  minor: Increment the minor version (0.X.0)"
    echo "  major: Increment the major version (X.0.0)"
    exit 1
}

# Check if a version increment type is provided
if [ $# -eq 0 ]; then
    usage
fi

# Validate the input
case $1 in
    patch|minor|major) ;;
    *) usage ;;
esac

# Ensure we're on the main branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "main" ]; then
    echo "Error: You must be on the main branch to create a release."
    exit 1
fi

# Ensure the working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean. Please commit or stash your changes."
    exit 1
fi

# Pull the latest changes
echo "Pulling latest changes..."
git pull origin main

# Bump the version
echo "Bumping $1 version..."
poetry version $1

# Get the new version
new_version=$(poetry version -s)

# Add pyproject.toml to git
git add pyproject.toml

# Commit the change
git commit -m "Release: Bump version to $new_version"

# Push the commit
echo "Pushing changes..."
git push origin main

echo "Release $new_version prepared and pushed."
echo "The CI/CD pipeline will create the release if configured correctly."