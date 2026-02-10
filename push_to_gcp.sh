#!/bin/bash

# --- CONFIGURATION ---
LOCAL_IMAGE="trailer-ocr:latest"
GCP_REGION="us-east5"
PROJECT_ID="rising-goal-461806-f9"
REPOSITORY="cloud-run-source-deploy"
GCP_IMAGE_NAME="trailer-ocr"

# Full path for GCP Artifact Registry
REMOTE_TAG="$GCP_REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$GCP_IMAGE_NAME:latest"

echo "Step 1: Authenticating Docker with Google Cloud..."
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev --quiet

echo "Step 2: Tagging local image $LOCAL_IMAGE to $REMOTE_TAG..."
docker tag $LOCAL_IMAGE $REMOTE_TAG

echo "Step 3: Pushing image to GCP..."
docker push $REMOTE_TAG

echo "---------------------------------------------------"
echo "Done! Your image is now in Artifact Registry."
