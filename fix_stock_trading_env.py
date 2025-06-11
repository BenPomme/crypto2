#!/usr/bin/env python3
"""
Script to generate the missing Railway environment variables for stock trading
"""

import json

# Load existing environment variables
with open('railway-env-variables-refined.json', 'r') as f:
    existing_vars = json.load(f)

# Required stock trading variables
stock_vars = {
    "ENABLE_STOCK_TRADING": "true",
    "STOCK_SYMBOLS": "SPY,QQQ,AAPL,TSLA",
    "ENABLE_SHORT_SELLING": "true",
    "STOCK_RISK_PER_TRADE": "0.01",
    "MAX_SHORT_EXPOSURE": "0.5"
}

# Firebase service account variables (from CLAUDE.md)
firebase_service_vars = {
    "FIREBASE_TYPE": "service_account",
    "FIREBASE_PRIVATE_KEY_ID": "YOUR_PRIVATE_KEY_ID",
    "FIREBASE_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----\\n...YOUR_KEY...\\n-----END PRIVATE KEY-----\\n",
    "FIREBASE_CLIENT_EMAIL": "YOUR_CLIENT_EMAIL@YOUR_PROJECT.iam.gserviceaccount.com",
    "FIREBASE_CLIENT_ID": "YOUR_CLIENT_ID",
    "FIREBASE_AUTH_URI": "https://accounts.google.com/o/oauth2/auth",
    "FIREBASE_TOKEN_URI": "https://oauth2.googleapis.com/token",
    "FIREBASE_AUTH_PROVIDER_CERT_URL": "https://www.googleapis.com/oauth2/v1/certs",
    "FIREBASE_CLIENT_CERT_URL": "YOUR_CERT_URL"
}

print("🔧 MISSING RAILWAY ENVIRONMENT VARIABLES FOR STOCK TRADING")
print("=" * 60)

print("\n📈 STOCK TRADING VARIABLES (REQUIRED):")
print("-" * 40)
for key, value in stock_vars.items():
    if key not in existing_vars:
        print(f"{key}={value}")

print("\n🔥 FIREBASE SERVICE ACCOUNT VARIABLES (CHECK IF MISSING):")
print("-" * 40)
print("Note: Replace placeholder values with actual Firebase service account credentials")
for key, value in firebase_service_vars.items():
    if key not in existing_vars:
        print(f"{key}={value}")

print("\n💡 INSTRUCTIONS:")
print("-" * 40)
print("1. Go to Railway dashboard: https://railway.app")
print("2. Select your project/service")
print("3. Go to Variables tab")
print("4. Add the missing variables shown above")
print("5. For Firebase variables, get the actual values from:")
print("   - Firebase Console > Project Settings > Service Accounts")
print("   - Generate new private key if needed")
print("6. After adding variables, Railway will automatically redeploy")

print("\n🚀 RAILWAY CLI COMMANDS (Alternative method):")
print("-" * 40)
print("# Set stock trading variables:")
for key, value in stock_vars.items():
    print(f'railway variables set {key}="{value}"')

print("\n# Verify variables are set:")
print("railway variables")

print("\n# Force redeploy after setting variables:")
print("railway redeploy")

print("\n⚠️  CURRENT STATUS:")
print("-" * 40)
print("❌ Stock trading is DISABLED because ENABLE_STOCK_TRADING is not set")
print("❌ Stock symbols are not configured")
print("❌ The bot will only trade crypto pairs until these variables are added")

# Create updated config file
updated_vars = existing_vars.copy()
updated_vars.update(stock_vars)

with open('railway-env-variables-with-stocks.json', 'w') as f:
    json.dump(updated_vars, f, indent=2)

print("\n✅ Created 'railway-env-variables-with-stocks.json' with stock trading enabled")
print("   (This file shows what the complete configuration should look like)")