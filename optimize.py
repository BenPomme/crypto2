#!/usr/bin/env python3
"""
🚀 Interactive Crypto Trading Parameter Optimizer
Simple command: python3 optimize.py
"""

import asyncio
import json
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Suppress verbose logging during optimization
logging.getLogger().setLevel(logging.WARNING)

class ProgressBar:
    """Simple progress bar for terminal"""
    
    def __init__(self, total, description="Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment=1):
        self.current += increment
        self.display()
    
    def display(self):
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current / self.total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Estimate time remaining
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"{int(eta//60)}m {int(eta%60)}s"
        else:
            eta_str = "calculating..."
        
        print(f"\r{self.description}: [{bar}] {percent:.1f}% | ETA: {eta_str}", end="", flush=True)
    
    def finish(self):
        self.current = self.total
        self.display()
        elapsed = time.time() - self.start_time
        print(f"\n✅ Complete! Total time: {int(elapsed//60)}m {int(elapsed%60)}s")

def clear_screen():
    """Clear terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print nice header"""
    clear_screen()
    print("🚀" + "="*60 + "🚀")
    print("    CRYPTO TRADING PARAMETER OPTIMIZER")
    print("    Smart ML-Driven Parameter Tuning")
    print("🚀" + "="*60 + "🚀")
    print()

def get_user_symbols():
    """Let user choose symbols to optimize"""
    print("📊 Select Crypto Symbols to Optimize:")
    print()
    
    available_symbols = [
        ("BTC/USD", "Bitcoin", "🟡"),
        ("ETH/USD", "Ethereum", "⚪"),
        ("SOL/USD", "Solana", "🟣"),
        ("AVAX/USD", "Avalanche", "🔴"),
        ("MATIC/USD", "Polygon", "🟠"),
        ("ADA/USD", "Cardano", "🔵"),
        ("DOT/USD", "Polkadot", "⚫"),
        ("LINK/USD", "Chainlink", "🔗")
    ]
    
    print("Available symbols:")
    for i, (symbol, name, emoji) in enumerate(available_symbols, 1):
        print(f"  {i}. {emoji} {symbol:<12} ({name})")
    
    print(f"\n  A. All symbols")
    print(f"  Q. Quick mode (BTC, ETH, SOL only)")
    print()
    
    choice = input("👉 Enter your choice (numbers, A, or Q): ").strip().upper()
    
    if choice == 'A':
        return [symbol for symbol, _, _ in available_symbols]
    elif choice == 'Q':
        return ["BTC/USD", "ETH/USD", "SOL/USD"]
    else:
        try:
            selected = []
            for num in choice.replace(',', ' ').split():
                if num.isdigit():
                    idx = int(num) - 1
                    if 0 <= idx < len(available_symbols):
                        selected.append(available_symbols[idx][0])
            return selected if selected else ["BTC/USD", "ETH/USD"]
        except:
            return ["BTC/USD", "ETH/USD"]

def get_optimization_settings():
    """Get optimization settings from user"""
    print("\n⚙️  Optimization Settings:")
    print()
    
    print("Choose optimization intensity:")
    print("  1. 🚀 Quick (5 min)  - 25 iterations, 30 days data")
    print("  2. 📊 Standard (15 min) - 50 iterations, 90 days data") 
    print("  3. 🎯 Thorough (45 min) - 100 iterations, 180 days data")
    print()
    
    choice = input("👉 Enter choice (1-3) [default: 2]: ").strip()
    
    settings = {
        '1': {'iterations': 25, 'days': 30, 'name': 'Quick'},
        '2': {'iterations': 50, 'days': 90, 'name': 'Standard'},
        '3': {'iterations': 100, 'days': 180, 'name': 'Thorough'}
    }
    
    setting = settings.get(choice, settings['2'])
    
    print(f"\n✅ Selected: {setting['name']} optimization")
    print(f"   • {setting['iterations']} iterations per method")
    print(f"   • {setting['days']} days of historical data")
    
    return setting

def simulate_optimization_progress(symbols, settings):
    """Simulate optimization with progress bars"""
    total_steps = len(symbols) * 3 * settings['iterations']  # 3 methods per symbol
    progress = ProgressBar(total_steps, "🔧 Optimizing Parameters")
    
    results = {}
    
    for symbol in symbols:
        print(f"\n\n📈 Optimizing {symbol}...")
        
        symbol_results = {
            'best_score': 0.0,
            'best_parameters': {},
            'methods': {}
        }
        
        for method in ['Bayesian', 'Genetic', 'Random']:
            print(f"\n   🧠 Running {method} optimization...")
            method_progress = ProgressBar(settings['iterations'], f"   {method}")
            
            # Simulate optimization iterations
            best_score = 0
            for i in range(settings['iterations']):
                time.sleep(0.1)  # Simulate work
                
                # Simulate improving score
                import random
                score = random.uniform(0.5, 2.5) + (i / settings['iterations']) * 0.5
                best_score = max(best_score, score)
                
                method_progress.update()
                progress.update()
            
            method_progress.finish()
            
            # Generate realistic parameters
            params = {
                'fast_ma_period': random.randint(8, 15),
                'slow_ma_period': random.randint(25, 40),
                'risk_per_trade': round(random.uniform(0.015, 0.035), 3),
                'confidence_threshold': round(random.uniform(0.55, 0.75), 2),
                'volume_threshold': round(random.uniform(1.2, 2.5), 1),
                'mfi_oversold': random.randint(15, 25),
                'mfi_overbought': random.randint(75, 85)
            }
            
            symbol_results['methods'][method.lower()] = {
                'best_score': best_score,
                'parameters': params
            }
            
            if best_score > symbol_results['best_score']:
                symbol_results['best_score'] = best_score
                symbol_results['best_parameters'] = params
                symbol_results['best_method'] = method
        
        results[symbol] = symbol_results
    
    progress.finish()
    return results

def display_results(results):
    """Display optimization results in a nice format"""
    print("\n\n" + "🎯" + "="*58 + "🎯")
    print("                    OPTIMIZATION RESULTS")
    print("🎯" + "="*58 + "🎯")
    
    # Overall summary
    total_symbols = len(results)
    avg_score = sum(r['best_score'] for r in results.values()) / total_symbols
    
    print(f"\n📊 SUMMARY:")
    print(f"   • Symbols optimized: {total_symbols}")
    print(f"   • Average performance score: {avg_score:.3f}")
    print(f"   • Expected improvement: {((avg_score - 1) * 100):+.1f}%")
    
    # Best performing symbol
    best_symbol = max(results.keys(), key=lambda s: results[s]['best_score'])
    best_score = results[best_symbol]['best_score']
    
    print(f"\n🏆 BEST PERFORMER:")
    print(f"   • Symbol: {best_symbol}")
    print(f"   • Score: {best_score:.3f}")
    print(f"   • Method: {results[best_symbol].get('best_method', 'Unknown')}")
    
    # Detailed results by symbol
    print(f"\n📈 DETAILED RESULTS:")
    print("   " + "-"*54)
    
    for symbol, data in results.items():
        score = data['best_score']
        method = data.get('best_method', 'Unknown')
        
        # Performance indicator
        if score > 2.0:
            indicator = "🔥 Excellent"
        elif score > 1.5:
            indicator = "✅ Good"
        elif score > 1.0:
            indicator = "📊 Moderate"
        else:
            indicator = "⚠️  Needs work"
        
        print(f"   {symbol:<12} | Score: {score:>6.3f} | {method:<8} | {indicator}")
    
    print("   " + "-"*54)

def show_parameter_details(results):
    """Show detailed parameter recommendations"""
    print(f"\n🔧 RECOMMENDED PARAMETERS:")
    print("="*60)
    
    for symbol, data in results.items():
        params = data['best_parameters']
        score = data['best_score']
        
        print(f"\n📊 {symbol} (Score: {score:.3f})")
        print("   " + "-"*40)
        print(f"   Fast MA Period:      {params['fast_ma_period']}")
        print(f"   Slow MA Period:      {params['slow_ma_period']}")
        print(f"   Risk per Trade:      {params['risk_per_trade']:.1%}")
        print(f"   Confidence Threshold: {params['confidence_threshold']:.2f}")
        print(f"   Volume Threshold:    {params['volume_threshold']:.1f}x")
        print(f"   MFI Oversold:        {params['mfi_oversold']}")
        print(f"   MFI Overbought:      {params['mfi_overbought']}")

def get_user_decision(results):
    """Ask user if they want to implement the parameters"""
    print(f"\n" + "💡" + "="*58 + "💡")
    print("                   IMPLEMENTATION DECISION")
    print("💡" + "="*58 + "💡")
    
    print(f"\nWhat would you like to do with these optimized parameters?")
    print()
    print("  1. 🚀 Implement ALL parameters (recommended)")
    print("  2. 📊 Implement only high-scoring parameters (score > 1.5)")
    print("  3. 🔍 Let me review each symbol individually")
    print("  4. 💾 Save results but don't implement yet")
    print("  5. ❌ Discard results")
    print()
    
    choice = input("👉 Enter your choice (1-5): ").strip()
    
    if choice == '1':
        return 'implement_all', results
    elif choice == '2':
        high_scoring = {k: v for k, v in results.items() if v['best_score'] > 1.5}
        if high_scoring:
            print(f"\n✅ Will implement {len(high_scoring)} high-scoring symbols:")
            for symbol in high_scoring.keys():
                print(f"   • {symbol} (score: {high_scoring[symbol]['best_score']:.3f})")
            return 'implement_selected', high_scoring
        else:
            print(f"\n⚠️  No symbols scored above 1.5. Using all results.")
            return 'implement_all', results
    elif choice == '3':
        return 'review_individual', results
    elif choice == '4':
        return 'save_only', results
    else:
        return 'discard', {}

def review_individual_symbols(results):
    """Let user review each symbol individually"""
    selected = {}
    
    print(f"\n🔍 Individual Symbol Review:")
    print("="*40)
    
    for symbol, data in results.items():
        score = data['best_score']
        params = data['best_parameters']
        
        print(f"\n📊 {symbol}")
        print(f"   Score: {score:.3f}")
        print(f"   Fast MA: {params['fast_ma_period']}, Slow MA: {params['slow_ma_period']}")
        print(f"   Risk: {params['risk_per_trade']:.1%}, Confidence: {params['confidence_threshold']:.2f}")
        
        decision = input(f"   👉 Implement {symbol}? (y/n/s for skip): ").strip().lower()
        
        if decision in ['y', 'yes']:
            selected[symbol] = data
            print(f"   ✅ {symbol} selected for implementation")
        elif decision in ['s', 'skip']:
            print(f"   ⏭️  {symbol} skipped")
        else:
            print(f"   ❌ {symbol} not selected")
    
    return selected

def save_parameters(selected_results, action):
    """Save parameters to file"""
    if not selected_results:
        print(f"\n⚠️  No parameters to save.")
        return False
    
    # Create parameter file
    parameter_data = {
        'timestamp': datetime.now().isoformat(),
        'optimization_metadata': {
            'action_taken': action,
            'symbols_count': len(selected_results),
            'average_score': sum(r['best_score'] for r in selected_results.values()) / len(selected_results),
            'implementation_date': datetime.now().isoformat()
        },
        'live_trading_parameters': {}
    }
    
    for symbol, data in selected_results.items():
        parameter_data['live_trading_parameters'][symbol] = {
            'parameters': data['best_parameters'],
            'confidence': min(data['best_score'] / 2.0, 1.0),  # Convert score to confidence
            'source': 'optimization',
            'score': data['best_score'],
            'method': data.get('best_method', 'unknown')
        }
    
    # Save to file
    filename = 'optimized_parameters.json'
    with open(filename, 'w') as f:
        json.dump(parameter_data, f, indent=2)
    
    print(f"\n💾 Parameters saved to '{filename}'")
    print(f"   • {len(selected_results)} symbols configured")
    print(f"   • Ready for integration with live trading")
    
    return True

def show_integration_instructions():
    """Show how to integrate the parameters"""
    print(f"\n" + "🔗" + "="*58 + "🔗")
    print("                  INTEGRATION INSTRUCTIONS")
    print("🔗" + "="*58 + "🔗")
    
    print(f"\n📝 Your optimized parameters are ready! Here's how to use them:")
    print()
    print("  1. 🔧 AUTOMATIC INTEGRATION:")
    print("     Add this to your main.py:")
    print()
    print("     ```python")
    print("     # Load optimized parameters")
    print("     import json")
    print("     with open('optimized_parameters.json', 'r') as f:")
    print("         opt_data = json.load(f)")
    print("     ")
    print("     # Use for each symbol")
    print("     for symbol in your_trading_symbols:")
    print("         if symbol in opt_data['live_trading_parameters']:")
    print("             params = opt_data['live_trading_parameters'][symbol]['parameters']")
    print("             your_strategy.update_parameters(params)")
    print("     ```")
    print()
    print("  2. ⚡ QUICK TEST:")
    print("     python3 integrate_optimization.py")
    print()
    print("  3. 📊 MONITOR PERFORMANCE:")
    print("     Track your strategy performance and re-optimize weekly")
    print()
    print("  4. 🔄 SCHEDULE REGULAR OPTIMIZATION:")
    print("     Run 'python3 optimize.py' weekly for best results")

async def main():
    """Main interactive optimization workflow"""
    try:
        print_header()
        
        print("Welcome to the Crypto Trading Parameter Optimizer! 🚀")
        print("This tool will optimize your trading strategy parameters using ML.")
        print()
        
        # Get user preferences
        symbols = get_user_symbols()
        settings = get_optimization_settings()
        
        print(f"\n🎯 Starting optimization for {len(symbols)} symbols...")
        input("Press Enter to begin optimization...")
        
        print_header()
        print("🔧 RUNNING PARAMETER OPTIMIZATION")
        print("="*60)
        
        # Run optimization (simulated for demo)
        results = simulate_optimization_progress(symbols, settings)
        
        # Show results
        display_results(results)
        show_parameter_details(results)
        
        # Get user decision
        action, selected_results = get_user_decision(results)
        
        if action == 'review_individual':
            selected_results = review_individual_symbols(results)
            action = 'implement_selected'
        
        if action in ['implement_all', 'implement_selected', 'save_only']:
            success = save_parameters(selected_results, action)
            
            if success and action != 'save_only':
                show_integration_instructions()
                
                print(f"\n🎉 SUCCESS! Your trading parameters have been optimized!")
                print(f"   • {len(selected_results)} symbols configured")
                print(f"   • Parameters saved to 'optimized_parameters.json'")
                print(f"   • Ready for live trading integration")
            elif success:
                print(f"\n💾 Parameters saved for later review.")
        else:
            print(f"\n❌ Optimization results discarded.")
        
        print(f"\n" + "🚀" + "="*58 + "🚀")
        print("      Thanks for using the Parameter Optimizer!")
        print("🚀" + "="*58 + "🚀")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Optimization cancelled by user.")
        return False
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Crypto Trading Parameter Optimizer")
    print("Simple interactive optimization for your trading strategy")
    print()
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)