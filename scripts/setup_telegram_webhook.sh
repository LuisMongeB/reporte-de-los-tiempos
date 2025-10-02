#!/bin/bash

# Script to set up Telegram webhook with secret token
# Usage: ./scripts/setup_telegram_webhook.sh <WEBHOOK_URL>

# Load environment variables
set -a
source .env.dev
set +a

# Check if URL is provided
if [ -z "$1" ]; then
    echo "❌ Error: Webhook URL not provided"
    echo "Usage: ./scripts/setup_telegram_webhook.sh <WEBHOOK_URL>"
    echo "Example: ./scripts/setup_telegram_webhook.sh https://abc123.ngrok-free.app"
    exit 1
fi

WEBHOOK_URL="$1/webhook/telegram"

echo "🔧 Setting up Telegram webhook..."
echo "📡 URL: $WEBHOOK_URL"
echo "🔐 Secret: ${TELEGRAM_WEBHOOK_SECRET:0:10}..." # Show first 10 chars only

# Set webhook with secret
RESPONSE=$(curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"${WEBHOOK_URL}\",
    \"secret_token\": \"${TELEGRAM_WEBHOOK_SECRET}\"
  }")

# Check if successful
if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "✅ Webhook set up successfully!"
    
    # Get webhook info
    echo ""
    echo "📋 Webhook Info:"
    curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo" | jq
else
    echo "❌ Failed to set webhook:"
    echo "$RESPONSE" | jq
    exit 1
fi
