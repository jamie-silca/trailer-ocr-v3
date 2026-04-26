# Deploying to Google Cloud Run

This service is configured to be easily deployed to Google Cloud Run using the "Continuously deploy from a repository" feature.

## Prerequisites

1.  **Google Cloud Project**: You need an active Google Cloud project with billing enabled.
2.  **Trailer Detector Service**: This service depends on the `trailer-detector` service. You must have the detector deployed and accessible via a public URL or internal Cloud Run URL.

## Deployment Steps

1.  **Push Changes**: Ensure your latest code (including the `Dockerfile` update) is pushed to your GitHub repository.
2.  **Go to Cloud Run**: Navigate to the [Cloud Run Console](https://console.cloud.google.com/run).
3.  **Create Service**: Click **Create Service**.
4.  **Source**: Select **Continuously deploy from a repository**.
5.  **Repository**: Connect your GitHub account and select your repository (`trailer-ocr-v3`).
6.  **Configuration**:
    *   **Region**: Select your preferred region (e.g., `us-central1`).
    *   **Authentication**: Choose "Allow unauthenticated invocations" if you want it publicly accessible, or restrict it as needed.
    *   **CPU allocation**: Select "CPU is only allocated during request processing" (default).
7.  **Container, Variables & Secrets**:
    *   Expand this section.
    *   **Environment variables**: Click **Add Variable**.
        *   Name: `DETECTOR_URL`
        *   Value: The URL of your deployed detector service (e.g., `https://detector-service-xyz.a.run.app`). **Do not include the trailing slash.**
    *   **Secret / variable** (EXP-25 portrait fallback — optional but recommended):
        *   Name: `OPENROUTER_API_KEY`
        *   Value: OpenRouter API key (`sk-or-v1-...`). Enables the Qwen3-VL fallback for portrait/stacked-vertical trailer-ID crops where PaddleOCR returns no text or a non-format-valid read. Without it, the service runs paddle-only.
        *   Recommended: store as a **Secret** in Secret Manager and mount as the env var, rather than pasting the raw value.
8.  **Create**: Click **Create**.

## Updating an existing service

Use this path when the service is already deployed and you only need to roll a new revision (e.g. to add `OPENROUTER_API_KEY` for the EXP-25 portrait fallback). New code on `main` is picked up by the continuous-deploy trigger automatically — these steps are about env-var / secret changes.

### Option A — Console

1.  Cloud Run → select **trailer-ocr** → **Edit & Deploy New Revision**.
2.  Open **Variables & Secrets**.
3.  Add `OPENROUTER_API_KEY`. Recommended: **Reference a Secret** (create the secret in Secret Manager first, then mount `latest`). Avoid pasting the raw key as a plain env var.
4.  Click **Deploy**.

### Option B — gcloud CLI

```bash
# One-time: store the key as a Secret Manager secret.
echo -n "sk-or-v1-..." | gcloud secrets create openrouter-api-key --data-file=-

# Grant the Cloud Run runtime service account access to the secret.
# Replace <runtime-sa> with the service's runtime SA, e.g.
#   PROJECT_NUMBER-compute@developer.gserviceaccount.com
gcloud secrets add-iam-policy-binding openrouter-api-key \
  --member=serviceAccount:<runtime-sa> \
  --role=roles/secretmanager.secretAccessor

# Roll a new revision with the secret mounted as the env var.
gcloud run services update trailer-ocr \
  --region us-east5 \
  --update-secrets=OPENROUTER_API_KEY=openrouter-api-key:latest
```

Verify by tailing the new revision's startup logs — `OcrProcessor._initialize` logs either `Qwen portrait fallback enabled (EXP-25).` or `OPENROUTER_API_KEY not set — Qwen portrait fallback disabled.`.

To rotate the key, add a new secret version (`gcloud secrets versions add openrouter-api-key --data-file=-`) and redeploy; the `:latest` mount picks it up on the next revision.

## Troubleshooting

-   **Port Issues**: The service utilizes the `PORT` environment variable injected by Cloud Run (defaulting to 8080). If the service fails to start, check the logs for port binding errors, though the `Dockerfile` is now configured to handle this automatically.
-   **Detector Connection**: If OCR works but detection fails, verify the `DETECTOR_URL` is correct and accessible from the Cloud Run service.
