#!/bin/bash
echo "=== QUICK RAILWAY STATUS CHECK ==="
echo ""
echo "1. Checking if bot is running (last 10 log lines):"
railway logs | tail -10

echo ""
echo "2. Checking for import errors:"
railway logs | grep -i "import\|module" | tail -5

echo ""
echo "3. Checking for ML parameter loading:"
railway logs | grep -i "optimized\|parameter" | tail -5

echo ""
echo "4. Checking recent trades/signals:"
railway logs | grep -i "signal\|trade" | tail -5

echo ""
echo "If you see recent logs above, the bot IS running and pydantic-settings IS installed on Railway."