# Deploying the Backend API to AWS

The backend is a stateless-except-for-session-cache FastAPI app (`src/api/app.py`)
packaged as a container (`Dockerfile` at repo root). Any AWS container service
that runs an image on a port works; **AWS App Runner** is the fastest path (no
VPC/load-balancer/cluster to configure) and is the recommended default below.
ECS Fargate and Elastic Beanstalk (Docker platform) are noted as alternatives —
they use the exact same image.

## 1. Build and push the image to ECR

```bash
# From the repo root
export AWS_REGION=us-east-1          # your region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO_NAME=document-delta-api

aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION" || true

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build -t "$REPO_NAME" .
docker tag "$REPO_NAME:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest"
```

## 2. Deploy on App Runner (recommended)

```bash
aws apprunner create-service \
  --service-name document-delta-api \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'"$AWS_ACCOUNT_ID"'.dkr.ecr.'"$AWS_REGION"'.amazonaws.com/'"$REPO_NAME"':latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentSecrets": {
          "NVIDIA_API_KEY": "arn:aws:secretsmanager:'"$AWS_REGION"':'"$AWS_ACCOUNT_ID"':secret:nvidia-api-key"
        },
        "RuntimeEnvironmentVariables": {
          "API_AUTH_TOKEN": "REPLACE_WITH_A_RANDOM_TOKEN",
          "API_CORS_ORIGINS": "https://<your-hf-space>.hf.space"
        }
      }
    },
    "AutoDeploymentsEnabled": true
  }' \
  --instance-configuration '{"Cpu": "1 vCPU", "Memory": "2 GB"}'
```

(Or use the console: **App Runner → Create service → Container registry → ECR
→** pick the image → set port `8000` → add the env vars/secrets above →
**Create & deploy**.) App Runner gives you an HTTPS URL immediately — that's
`API_BASE_URL` for the frontend.

Storing `NVIDIA_API_KEY` in **Secrets Manager** (referenced above) rather than
a plain env var is the safer default since this is a publicly reachable
service; a plain `RuntimeEnvironmentVariables` entry works too if you'd rather
skip Secrets Manager for a quick demo.

## 3. Alternatives (same image)

- **ECS Fargate**: push the same image, define a task definition (port 8000,
  same env vars/secrets), run it behind an Application Load Balancer. More
  setup (VPC, ALB, target group) but gives you full control over scaling and
  networking.
- **Elastic Beanstalk (Docker platform)**: `eb init` / `eb create` pointing at
  this `Dockerfile` — Beanstalk builds and runs it directly, no ECR push
  needed if you deploy from source.

## 4. Required environment variables

| Variable | Required | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | Yes, for real inference | Server-side only — never sent to or stored by the frontend. Without it, `/chat` still works but returns the offline fallback. |
| `API_AUTH_TOKEN` | Recommended | If set, every request needs `Authorization: Bearer <token>`. This API triggers billable NVIDIA API calls per `/chat` request — leaving it open on a public URL is a cost risk, not just a security one. |
| `API_CORS_ORIGINS` | Recommended | Comma-separated origins allowed to call the API from a browser (your deployed Streamlit frontend's URL). Defaults to `*` (any origin) if unset — fine for local dev, not for production. |

Everything in `.env.example` (LLM model, thresholds, etc.) can also be set as
environment variables the same way.

## 5. Verify

```bash
curl https://<app-runner-url>/health
# {"status":"ok"}

curl -X POST https://<app-runner-url>/delta \
  -H "Authorization: Bearer <API_AUTH_TOKEN>" \
  -F "sample_pair=pair_02"
```

## Known limitation: in-memory session state

`src/api/sessions.py` keeps session state (ingested documents, delta results,
ChromaDB index handles) in a single process's memory, evicted after
`API_SESSION_TTL_SECONDS` (default 1 hour). This matches the CLI's own
single-run scope and is fine for a single App Runner instance. If you scale to
multiple instances/replicas, a request can land on an instance that doesn't
have that session's state — the fix is moving session state to a shared store
(Redis for the in-memory bits, S3 for the ChromaDB persist directory) instead
of per-process memory. Documented here rather than silently broken behind
auto-scaling.
