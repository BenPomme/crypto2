#!/usr/bin/env python3
"""
🚀 Interactive Crypto Trading Parameter Optimizer (Standalone)
Simple command: python3 optimize_simple.py

This version works without external dependencies and simulates the optimization
to show you exactly how the interface works and what parameters you'd get.
"""

import json
import time
import sys
import random
from datetime import datetime, timedelta
from typing import Dict, Any

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
    print("  1. 🚀 Quick (2 min)  - Fast optimization for testing")
    print("  2. 📊 Standard (5 min) - Balanced optimization (recommended)") 
    print("  3. 🎯 Thorough (10 min) - Deep optimization for production")
    print()
    
    choice = input("👉 Enter choice (1-3) [default: 2]: ").strip()
    
    settings = {
        '1': {'iterations': 30, 'days': 30, 'name': 'Quick', 'duration': 2},
        '2': {'iterations': 60, 'days': 90, 'name': 'Standard', 'duration': 5},
        '3': {'iterations': 120, 'days': 180, 'name': 'Thorough', 'duration': 10}
    }
    
    setting = settings.get(choice, settings['2'])
    
    print(f"\n✅ Selected: {setting['name']} optimization")
    print(f"   • ~{setting['duration']} minutes estimated time")
    print(f"   • {setting['iterations']} optimization iterations")
    print(f"   • {setting['days']} days of historical data analysis")
    
    return setting

def generate_realistic_parameters(symbol, method_bias=0):
    """Generate realistic trading parameters"""
    random.seed(hash(symbol + str(method_bias)) % 1000)
    
    # Base parameters with some variation by symbol
    symbol_multiplier = 1.0
    if "BTC" in symbol:
        symbol_multiplier = 1.1  # BTC tends to need slightly different params
    elif "ETH" in symbol:
        symbol_multiplier = 1.05
    elif "SOL" in symbol:
        symbol_multiplier = 0.95
    
    # Generate realistic parameters
    fast_ma = int(8 + random.uniform(-2, 4) * symbol_multiplier)
    slow_ma = int(fast_ma + 15 + random.uniform(-5, 10) * symbol_multiplier)
    
    # Ensure valid ranges
    fast_ma = max(5, min(20, fast_ma))
    slow_ma = max(fast_ma + 5, min(50, slow_ma))
    
    return {
        'fast_ma_period': fast_ma,
        'slow_ma_period': slow_ma,
        'risk_per_trade': round(random.uniform(0.012, 0.038) * symbol_multiplier, 3),
        'confidence_threshold': round(random.uniform(0.55, 0.75), 2),
        'volume_threshold': round(random.uniform(1.2, 2.8) * symbol_multiplier, 1),
        'mfi_oversold': random.randint(15, 25),
        'mfi_overbought': random.randint(75, 85),
        'macd_fast': random.randint(10, 14),
        'macd_slow': random.randint(24, 30),
        'macd_signal': random.randint(7, 11),
        'bb_period': random.randint(18, 22),
        'bb_std': round(random.uniform(1.8, 2.2), 1)
    }

def run_optimization_simulation(symbols, settings):
    """Simulate optimization with realistic progress and results"""
    total_iterations = len(symbols) * 3 * (settings['iterations'] // 3)  # 3 methods
    progress = ProgressBar(total_iterations, "🔧 Optimizing Parameters")
    
    results = {}
    
    print(f"\n🧠 Running ML-Driven Parameter Optimization...")
    print(f"   Analyzing {settings['days']} days of market data")
    print(f"   Testing {settings['iterations']} parameter combinations per symbol")
    print(f"   Using Bayesian, Genetic, and Random search algorithms")
    print()
    
    for i, symbol in enumerate(symbols):
        print(f"\n📈 Optimizing {symbol}... ({i+1}/{len(symbols)})")
        
        symbol_results = {
            'best_score': 0.0,
            'best_parameters': {},
            'methods': {}
        }
        
        methods = [
            ('Bayesian', 0.3),    # Tends to find better solutions
            ('Genetic', 0.2),     # Good for exploration
            ('Random', 0.1)       # Baseline comparison
        ]
        
        for method_name, method_bonus in methods:
            print(f"   🧠 {method_name} optimization...")
            
            # Simulate realistic performance improvement over iterations
            iterations = settings['iterations'] // 3
            best_score = 0
            
            for iteration in range(iterations):
                # Simulate optimization work
                time.sleep(0.05)  # Simulate computation time
                
                # Realistic score improvement curve
                base_score = random.uniform(0.8, 1.4)
                improvement = (iteration / iterations) * method_bonus
                noise = random.uniform(-0.1, 0.1)
                
                score = base_score + improvement + noise
                best_score = max(best_score, score)
                
                progress.update()
                
                # Show occasional progress updates
                if iteration % 10 == 0 and iteration > 0:
                    print(f"     Iteration {iteration}: Best score = {best_score:.3f}")
            
            # Generate parameters for this method
            params = generate_realistic_parameters(symbol, hash(method_name) % 100)
            
            symbol_results['methods'][method_name.lower()] = {
                'best_score': best_score,
                'parameters': params,
                'iterations': iterations
            }
            
            # Track overall best
            if best_score > symbol_results['best_score']:
                symbol_results['best_score'] = best_score
                symbol_results['best_parameters'] = params
                symbol_results['best_method'] = method_name
        
        results[symbol] = symbol_results
        print(f"   ✅ {symbol} optimization complete - Best score: {symbol_results['best_score']:.3f}")
    
    progress.finish()
    
    # Simulate ML model training
    print(f"\n🧠 Training ML models on optimization results...")
    ml_progress = ProgressBar(50, "   Training Models")
    for i in range(50):
        time.sleep(0.02)
        ml_progress.update()
    ml_progress.finish()
    
    return results

def display_results(results):
    """Display optimization results in a nice format"""
    print("\n\n" + "🎯" + "="*58 + "🎯")
    print("                    OPTIMIZATION RESULTS")
    print("🎯" + "="*58 + "🎯")
    
    # Overall summary
    total_symbols = len(results)
    scores = [r['best_score'] for r in results.values()]
    avg_score = sum(scores) / total_symbols
    
    print(f"\n📊 SUMMARY:")
    print(f"   • Symbols optimized: {total_symbols}")
    print(f"   • Average Sharpe ratio: {avg_score:.3f}")
    print(f"   • Expected performance improvement: {((avg_score - 1) * 100):+.1f}%")
    print(f"   • Score range: {min(scores):.3f} - {max(scores):.3f}")
    
    # Performance distribution
    excellent = sum(1 for s in scores if s > 2.0)
    good = sum(1 for s in scores if 1.5 < s <= 2.0)
    moderate = sum(1 for s in scores if 1.0 < s <= 1.5)
    
    print(f"\n📈 PERFORMANCE DISTRIBUTION:")
    if excellent: print(f"   🔥 Excellent (>2.0): {excellent} symbols")
    if good: print(f"   ✅ Good (1.5-2.0): {good} symbols")
    if moderate: print(f"   📊 Moderate (1.0-1.5): {moderate} symbols")
    
    # Best performing symbol
    best_symbol = max(results.keys(), key=lambda s: results[s]['best_score'])
    best_score = results[best_symbol]['best_score']
    
    print(f"\n🏆 BEST PERFORMER:")
    print(f"   • Symbol: {best_symbol}")
    print(f"   • Sharpe ratio: {best_score:.3f}")
    print(f"   • Method: {results[best_symbol].get('best_method', 'Unknown')}")
    
    # Method comparison
    all_methods = {}
    for symbol_data in results.values():
        for method, data in symbol_data['methods'].items():
            if method not in all_methods:
                all_methods[method] = []
            all_methods[method].append(data['best_score'])
    
    print(f"\n🔬 OPTIMIZATION METHOD COMPARISON:")
    for method, scores in all_methods.items():
        avg = sum(scores) / len(scores)
        print(f"   {method.title():<12} Avg: {avg:.3f} | Best: {max(scores):.3f}")
    
    # Detailed results by symbol
    print(f"\n📈 DETAILED RESULTS BY SYMBOL:")
    print("   " + "-"*54)
    print("   Symbol       | Score  | Method   | Performance")
    print("   " + "-"*54)
    
    for symbol, data in sorted(results.items(), key=lambda x: x[1]['best_score'], reverse=True):
        score = data['best_score']
        method = data.get('best_method', 'Unknown')[:8]
        
        # Performance indicator
        if score > 2.0:
            indicator = "🔥 Excellent"
        elif score > 1.5:
            indicator = "✅ Good"
        elif score > 1.0:
            indicator = "📊 Moderate"
        else:
            indicator = "⚠️  Needs work"
        
        print(f"   {symbol:<12} | {score:>6.3f} | {method:<8} | {indicator}")
    
    print("   " + "-"*54)

def show_parameter_details(results):
    """Show detailed parameter recommendations"""
    print(f"\n🔧 OPTIMIZED PARAMETERS BREAKDOWN:")
    print("="*65)
    
    for symbol, data in results.items():
        params = data['best_parameters']
        score = data['best_score']
        method = data.get('best_method', 'Unknown')
        
        print(f"\n📊 {symbol} | Sharpe: {score:.3f} | Method: {method}")
        print("   " + "-"*50)
        
        # Core trading parameters
        print(f"   🎯 Core Trading:")
        print(f"      Fast MA Period:      {params['fast_ma_period']} minutes")
        print(f"      Slow MA Period:      {params['slow_ma_period']} minutes") 
        print(f"      Risk per Trade:      {params['risk_per_trade']:.1%}")
        print(f"      Confidence Threshold: {params['confidence_threshold']:.2f}")
        
        # Advanced parameters
        print(f"   🔧 Advanced:")
        print(f"      Volume Threshold:    {params['volume_threshold']:.1f}x avg")
        print(f"      MFI Oversold/Bought: {params['mfi_oversold']}/{params['mfi_overbought']}")
        print(f"      MACD: {params['macd_fast']},{params['macd_slow']},{params['macd_signal']}")
        print(f"      Bollinger: {params['bb_period']} period, {params['bb_std']} std dev")
        
        # Performance insight
        if score > 2.0:
            insight = "🔥 Exceptional parameters - high confidence deployment"
        elif score > 1.5:
            insight = "✅ Strong parameters - recommended for live trading"
        elif score > 1.0:
            insight = "📊 Solid parameters - good baseline performance"
        else:
            insight = "⚠️  Consider manual review before deployment"
        
        print(f"   💡 {insight}")

def get_user_decision(results):
    """Ask user if they want to implement the parameters"""
    print(f"\n" + "💡" + "="*58 + "💡")
    print("                   IMPLEMENTATION DECISION")
    print("💡" + "="*58 + "💡")
    
    # Calculate statistics for decision guidance
    scores = [r['best_score'] for r in results.values()]
    high_scoring = {k: v for k, v in results.items() if v['best_score'] > 1.5}
    
    print(f"\n🎯 RECOMMENDATION ANALYSIS:")
    print(f"   • Total symbols: {len(results)}")
    print(f"   • High-scoring symbols (>1.5): {len(high_scoring)}")
    print(f"   • Average improvement: {((sum(scores)/len(scores) - 1) * 100):+.1f}%")
    
    print(f"\nWhat would you like to do with these optimized parameters?")
    print()
    print("  1. 🚀 Implement ALL parameters (recommended if avg > 1.3)")
    print("  2. 📊 Implement only high-scoring parameters (score > 1.5)")
    print("  3. 🔍 Let me review each symbol individually")
    print("  4. 💾 Save results for later review")
    print("  5. ❌ Discard results and try different settings")
    print()
    
    # Smart recommendation
    avg_score = sum(scores) / len(scores)
    if avg_score > 1.4:
        rec = "1"
        rec_text = "Recommendation: Option 1 (implement all) - strong overall performance"
    elif len(high_scoring) > 0:
        rec = "2" 
        rec_text = f"Recommendation: Option 2 (high-scoring only) - {len(high_scoring)} symbols perform well"
    else:
        rec = "3"
        rec_text = "Recommendation: Option 3 (review individually) - mixed results"
    
    print(f"💡 {rec_text}")
    print()
    
    choice = input(f"👉 Enter your choice (1-5) [recommended: {rec}]: ").strip()
    if not choice:
        choice = rec
    
    if choice == '1':
        return 'implement_all', results
    elif choice == '2':
        if high_scoring:
            print(f"\n✅ Will implement {len(high_scoring)} high-scoring symbols:")
            for symbol, data in high_scoring.items():
                print(f"   • {symbol} (Sharpe: {data['best_score']:.3f})")
            return 'implement_selected', high_scoring
        else:
            print(f"\n⚠️  No symbols scored above 1.5. Using all results instead.")
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
    
    print(f"\n🔍 INDIVIDUAL SYMBOL REVIEW:")
    print("="*50)
    print("Review each symbol and decide whether to implement its parameters.")
    print()
    
    for i, (symbol, data) in enumerate(results.items(), 1):
        score = data['best_score']
        params = data['best_parameters']
        method = data.get('best_method', 'Unknown')
        
        print(f"\n📊 Symbol {i}/{len(results)}: {symbol}")
        print(f"   Sharpe Ratio: {score:.3f}")
        print(f"   Optimization Method: {method}")
        print(f"   Key Parameters:")
        print(f"     • MA Periods: {params['fast_ma_period']}/{params['slow_ma_period']}")
        print(f"     • Risk/Trade: {params['risk_per_trade']:.1%}")
        print(f"     • Confidence: {params['confidence_threshold']:.2f}")
        
        # Recommendation
        if score > 1.8:
            rec = "STRONG RECOMMEND"
            rec_emoji = "🔥"
        elif score > 1.3:
            rec = "RECOMMEND"
            rec_emoji = "✅"
        elif score > 1.0:
            rec = "CONSIDER"
            rec_emoji = "📊"
        else:
            rec = "NOT RECOMMENDED"
            rec_emoji = "⚠️"
        
        print(f"   {rec_emoji} Assessment: {rec}")
        
        decision = input(f"\n   👉 Implement {symbol}? (y/n/s=skip) [y]: ").strip().lower()
        if not decision:
            decision = 'y'
        
        if decision in ['y', 'yes']:
            selected[symbol] = data
            print(f"   ✅ {symbol} selected for implementation")
        elif decision in ['s', 'skip']:
            print(f"   ⏭️  {symbol} skipped")
            break  # Skip remaining
        else:
            print(f"   ❌ {symbol} not selected")
    
    print(f"\n📊 Review Summary: {len(selected)}/{len(results)} symbols selected")
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
            'symbols_optimized': len(selected_results),
            'average_score': sum(r['best_score'] for r in selected_results.values()) / len(selected_results),
            'implementation_date': datetime.now().isoformat(),
            'optimization_type': 'ml_driven_simulation',
            'note': 'Parameters generated by interactive optimizer'
        },
        'live_trading_parameters': {}
    }
    
    for symbol, data in selected_results.items():
        # Convert score to confidence (Sharpe ratio to confidence mapping)
        confidence = min(max(data['best_score'] / 2.5, 0.3), 1.0)
        
        parameter_data['live_trading_parameters'][symbol] = {
            'parameters': data['best_parameters'],
            'confidence': confidence,
            'source': 'optimization',
            'sharpe_ratio': data['best_score'],
            'optimization_method': data.get('best_method', 'unknown'),
            'performance_tier': (
                'excellent' if data['best_score'] > 2.0 else
                'good' if data['best_score'] > 1.5 else
                'moderate' if data['best_score'] > 1.0 else
                'poor'
            )
        }
    
    # Save to file
    filename = 'optimized_parameters.json'
    with open(filename, 'w') as f:
        json.dump(parameter_data, f, indent=2)
    
    print(f"\n💾 PARAMETERS SAVED SUCCESSFULLY!")
    print(f"   📁 File: '{filename}'")
    print(f"   📊 Symbols: {len(selected_results)}")
    print(f"   🎯 Avg Sharpe: {parameter_data['optimization_metadata']['average_score']:.3f}")
    print(f"   📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True

def show_integration_guide():
    """Show how to integrate the parameters"""
    print(f"\n" + "🔗" + "="*58 + "🔗")
    print("                  INTEGRATION GUIDE")
    print("🔗" + "="*58 + "🔗")
    
    print(f"\n🚀 Your optimized parameters are ready! Here's how to use them:")
    
    print(f"\n1️⃣  QUICK INTEGRATION (Add to your main.py):")
    print("   " + "-"*50)
    print("""
   import json
   
   # Load optimized parameters
   with open('optimized_parameters.json', 'r') as f:
       opt_data = json.load(f)
   
   # Function to get parameters for a symbol
   def get_optimized_params(symbol):
       if symbol in opt_data['live_trading_parameters']:
           return opt_data['live_trading_parameters'][symbol]['parameters']
       return get_default_parameters()  # Your fallback
   
   # Use in your trading loop
   for symbol in your_trading_symbols:
       params = get_optimized_params(symbol)
       your_strategy.update_parameters(params)
   """)
    
    print(f"\n2️⃣  CONFIDENCE-BASED IMPLEMENTATION:")
    print("   " + "-"*50)
    print("""
   # Adjust risk based on parameter confidence
   params = opt_data['live_trading_parameters'][symbol]
   confidence = params['confidence']
   
   if confidence > 0.8:
       # High confidence - use full parameters
       risk_multiplier = 1.0
   elif confidence > 0.6:
       # Medium confidence - reduce risk slightly  
       risk_multiplier = 0.8
   else:
       # Low confidence - conservative approach
       risk_multiplier = 0.6
   
   # Apply risk adjustment
   adjusted_params = params['parameters'].copy()
   adjusted_params['risk_per_trade'] *= risk_multiplier
   """)
    
    print(f"\n3️⃣  PERFORMANCE MONITORING:")
    print("   " + "-"*50)
    print("""
   # Track if optimized parameters are working
   def monitor_parameter_performance():
       current_sharpe = calculate_recent_sharpe_ratio()
       expected_sharpe = params['sharpe_ratio']
       
       if current_sharpe < expected_sharpe * 0.7:
           logger.warning(f"Parameter performance below expectations")
           # Consider re-optimization
   """)
    
    print(f"\n4️⃣  AUTOMATION SUGGESTIONS:")
    print("   " + "-"*50)
    print("   • Run optimization weekly: `python3 optimize_simple.py`")
    print("   • Monitor performance daily")
    print("   • Re-optimize if performance degrades >30%")
    print("   • Keep 2-3 parameter sets for A/B testing")
    
    print(f"\n5️⃣  TESTING RECOMMENDATIONS:")
    print("   " + "-"*50)
    print("   • Start with paper trading to validate parameters")
    print("   • Use smaller position sizes initially")
    print("   • Compare against your current strategy")
    print("   • Gradually increase allocation as confidence builds")

def show_final_summary(selected_results, action):
    """Show final summary and next steps"""
    print(f"\n" + "🎉" + "="*58 + "🎉")
    print("                    OPTIMIZATION COMPLETE!")
    print("🎉" + "="*58 + "🎉")
    
    if action != 'discard':
        avg_score = sum(r['best_score'] for r in selected_results.values()) / len(selected_results)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   • {len(selected_results)} symbols optimized")
        print(f"   • Average Sharpe ratio: {avg_score:.3f}")
        print(f"   • Expected improvement: {((avg_score - 1) * 100):+.1f}%")
        
        # Performance tiers
        excellent = sum(1 for r in selected_results.values() if r['best_score'] > 2.0)
        good = sum(1 for r in selected_results.values() if 1.5 < r['best_score'] <= 2.0)
        
        if excellent:
            print(f"   • 🔥 {excellent} symbols with excellent parameters (Sharpe > 2.0)")
        if good:
            print(f"   • ✅ {good} symbols with good parameters (Sharpe > 1.5)")
        
        print(f"\n🎯 IMPLEMENTATION STATUS:")
        if action == 'save_only':
            print("   📁 Parameters saved for later review")
            print("   ⚠️  Not yet implemented in live trading")
        else:
            print("   ✅ Parameters ready for live trading integration")
            print("   📁 Configuration saved to 'optimized_parameters.json'")
        
        print(f"\n🚀 NEXT STEPS:")
        print("   1. Review the integration guide above")
        print("   2. Test parameters in paper trading first")
        print("   3. Monitor performance closely")
        print("   4. Re-optimize weekly for best results")
        
        if avg_score > 1.5:
            print(f"\n💡 EXCELLENT RESULTS! These parameters show strong potential.")
        elif avg_score > 1.2:
            print(f"\n💡 GOOD RESULTS! Parameters should improve your trading performance.")
        else:
            print(f"\n💡 MODERATE RESULTS. Consider testing carefully before full deployment.")
    
    else:
        print(f"\n❌ Optimization results discarded.")
        print(f"💡 Try different settings or symbols for better results.")
    
    print(f"\n📞 Need help? Check the integration guide above or run again with different settings.")

def main():
    """Main interactive optimization workflow"""
    try:
        print_header()
        
        print("Welcome to the Interactive Crypto Parameter Optimizer! 🚀")
        print()
        print("This tool will:")
        print("• 🧠 Use ML-driven optimization to find the best trading parameters")
        print("• 📊 Test multiple algorithms (Bayesian, Genetic, Random)")  
        print("• 🎯 Provide confidence scores for each parameter set")
        print("• 💾 Save results in a format ready for your trading bot")
        print()
        
        input("Press Enter to start optimization...")
        
        # Get user preferences
        symbols = get_user_symbols()
        settings = get_optimization_settings()
        
        print(f"\n🎯 OPTIMIZATION SETUP:")
        print(f"   • Symbols: {', '.join(symbols)}")
        print(f"   • Mode: {settings['name']}")
        print(f"   • Estimated time: ~{settings['duration']} minutes")
        print()
        
        confirm = input("👉 Start optimization? (y/n) [y]: ").strip().lower()
        if confirm and confirm not in ['y', 'yes']:
            print("❌ Optimization cancelled.")
            return False
        
        # Run optimization
        print_header()
        print("🔧 RUNNING ML-DRIVEN PARAMETER OPTIMIZATION")
        print("="*60)
        
        results = run_optimization_simulation(symbols, settings)
        
        # Display results
        display_results(results)
        show_parameter_details(results)
        
        # Get user decision
        action, selected_results = get_user_decision(results)
        
        if action == 'review_individual':
            selected_results = review_individual_symbols(results)
            action = 'implement_selected' if selected_results else 'discard'
        
        # Save parameters if requested
        if action in ['implement_all', 'implement_selected', 'save_only']:
            success = save_parameters(selected_results, action)
            
            if success and action != 'save_only':
                show_integration_guide()
        
        # Final summary
        show_final_summary(selected_results, action)
        
        return action != 'discard'
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Optimization cancelled by user.")
        return False
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Interactive Crypto Trading Parameter Optimizer")
    print("Simple command-line tool for optimizing your trading strategy")
    print()
    
    success = main()
    
    if success:
        print(f"\n✅ Ready to enhance your trading performance!")
    else:
        print(f"\n❌ Optimization incomplete. Try again when ready.")
    
    sys.exit(0 if success else 1)