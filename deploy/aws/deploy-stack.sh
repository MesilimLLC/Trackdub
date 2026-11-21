#!/usr/bin/env bash
# Creates or updates the POC network, bucket and instance role. Safe to re-run.
#
#   deploy/aws/deploy-stack.sh                      # stack snapsoft-poc in us-east-1
#   deploy/aws/deploy-stack.sh --stack acme-poc --bucket-name acme-trackdub-poc
set -euo pipefail

STACK=snapsoft-poc
REGION=us-east-1
BUCKET_NAME=""
TOKEN_PARAM=/trackdub/ghcr-read-token
VPC_CIDR=10.77.0.0/16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack) STACK="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --bucket-name) BUCKET_NAME="$2"; shift 2 ;;
    --token-parameter) TOKEN_PARAM="$2"; shift 2 ;;
    --vpc-cidr) VPC_CIDR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: deploy-stack.sh [--stack NAME] [--region REGION] [--bucket-name NAME] [--token-parameter /path] [--vpc-cidr CIDR]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ${#STACK} -gt 42 ]]; then
  echo "error: stack name must be 42 characters or fewer (IAM role names are limited to 64)." >&2
  exit 1
fi

TEMPLATE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/snapsoft-poc.yaml"

aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK" \
  --template-file "$TEMPLATE" \
  --capabilities CAPABILITY_NAMED_IAM \
  --tags Project=trackdub-poc-e2e \
  --parameter-overrides \
    "VpcCidr=$VPC_CIDR" \
    "BucketName=$BUCKET_NAME" \
    "TokenParameterName=$TOKEN_PARAM"

aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table

echo
echo "Next: store the GHCR token at $TOKEN_PARAM if it is not there yet (see docs/handoff/snapsoft-quickstart.md)."
