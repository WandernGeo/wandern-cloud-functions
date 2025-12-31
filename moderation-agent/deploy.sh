#!/bin/bash
set -e

# Configuration
FUNCTION_NAME="wandern-moderation-agent"
REGION="us-central1"
PROJECT_ID="wandern-project-startup"

# Load/Check Environment Variables
if [ -f ../../wandern-back/.env ]; then
  export $(cat ../../wandern-back/.env | grep GOOGLE_API_KEY | xargs)
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ Error: GOOGLE_API_KEY is not set. Please ensure it is in ../../wandern-back/.env"
    exit 1
fi

echo "🚀 Deploying $FUNCTION_NAME to $REGION..."

gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=python311 \
    --region=$REGION \
    --source=. \
    --entry-point=moderate_content \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars GOOGLE_API_KEY=$GOOGLE_API_KEY \
    --project=$PROJECT_ID

echo "✅ Deployment complete!"
