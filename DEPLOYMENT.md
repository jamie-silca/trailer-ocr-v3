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
8.  **Create**: Click **Create**.

## Troubleshooting

-   **Port Issues**: The service utilizes the `PORT` environment variable injected by Cloud Run (defaulting to 8080). If the service fails to start, check the logs for port binding errors, though the `Dockerfile` is now configured to handle this automatically.
-   **Detector Connection**: If OCR works but detection fails, verify the `DETECTOR_URL` is correct and accessible from the Cloud Run service.
