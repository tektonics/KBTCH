# simulation_analyzer.py
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

class SimulationAnalyzer:
    """Analyze simulation results and generate performance reports"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.data = self._load_simulation_data()
        
    def _load_simulation_data(self) -> Dict:
        """Load simulation data from JSON file"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Could not load simulation data: {e}")
    
    def print_summary_report(self):
        """Print a comprehensive summary report"""
        summary = self.data['summary']
        trades = self.data['trades']
        positions = self.data.get('final_positions', {})
        
        print("=" * 70)
        print("SIMULATION PERFORMANCE REPORT")
        print("=" * 70)
        
        # Overall Performance
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"Starting Balance:     ${summary['starting_balance']:>10,.2f}")
        print(f"Final Value:          ${summary['total_value']:>10,.2f}")
        print(f"Total Return:         ${summary['total_value'] - summary['starting_balance']:>+10,.2f}")
        print(f"Return %:             {summary['total_return_pct']:>+10.2f}%")
        print(f"Max Drawdown:         {summary['max_drawdown_pct']:>10.2f}%")
        
        # Trading Activity
        print(f"\n📈 TRADING ACTIVITY")
        print(f"Total Trades:         {len(trades):>10}")
        print(f"Completed Trades:     {summary['completed_trades']:>10}")
        print(f"Open Positions:       {summary['open_positions']:>10}")
        print(f"Realized P&L:         ${summary['realized_pnl']:>+10,.2f}")
        print(f"Unrealized P&L:       ${summary['unrealized_pnl']:>+10,.2f}")
        
        # Trade Analysis
        if trades:
            self._analyze_trades(trades)
        
        # Position Analysis
        if positions:
            self._analyze_positions(positions)
        
        # Risk Metrics
        self._calculate_risk_metrics(trades, summary)
    
    def _analyze_trades(self, trades: List[Dict]):
        """Analyze individual trades"""
        opens = [t for t in trades if t['type'] == 'OPEN']
        closes = [t for t in trades if t['type'] == 'CLOSE']
        
        print(f"\n🔄 TRADE BREAKDOWN")
        print(f"Opening Trades:       {len(opens):>10}")
        print(f"Closing Trades:       {len(closes):>10}")
        
        if opens:
            avg_position_size = np.mean([t['cost'] for t in opens])
            print(f"Avg Position Size:    ${avg_position_size:>10,.2f}")
        
        if closes:
            pnls = [t['pnl'] for t in closes]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            
            print(f"\n💰 TRADE OUTCOMES")
            print(f"Winning Trades:       {len(winning_trades):>10}")
            print(f"Losing Trades:        {len(losing_trades):>10}")
            
            if len(winning_trades) + len(losing_trades) > 0:
                win_rate = len(winning_trades) / len(closes) * 100
                print(f"Win Rate:             {win_rate:>10.1f}%")
            
            if winning_trades:
                avg_win = np.mean(winning_trades)
                print(f"Average Win:          ${avg_win:>+10,.2f}")
            
            if losing_trades:
                avg_loss = np.mean(losing_trades)
                print(f"Average Loss:         ${avg_loss:>+10,.2f}")
            
            if winning_trades and losing_trades:
                profit_factor = sum(winning_trades) / abs(sum(losing_trades))
                print(f"Profit Factor:        {profit_factor:>10.2f}")
    
    def _analyze_positions(self, positions: Dict):
        """Analyze final positions"""
        print(f"\n🎯 FINAL POSITIONS")
        
        for ticker, pos in positions.items():
            print(f"{ticker[:15]:15} | "
                  f"{pos['side']:3} | "
                  f"Qty: {pos['quantity']:3} | "
                  f"Entry: ${pos['entry_price']:6.0f} | "
                  f"Current: ${pos['current_price'] or 0:6.0f} | "
                  f"P&L: ${pos['unrealized_pnl']:+8.2f}")
    
    def _calculate_risk_metrics(self, trades: List[Dict], summary: Dict):
        """Calculate additional risk metrics"""
        print(f"\n⚖️  RISK METRICS")
        
        closes = [t for t in trades if t['type'] == 'CLOSE']
        if not closes:
            print("No completed trades for risk analysis")
            return
        
        pnls = [t['pnl'] for t in closes]
        
        # Sharpe-like ratio (simplified)
        if len(pnls) > 1:
            mean_return = np.mean(pnls)
            std_return = np.std(pnls, ddof=1)
            if std_return > 0:
                sharpe_like = mean_return / std_return
                print(f"Return/Risk Ratio:    {sharpe_like:>10.2f}")
        
        # Largest win/loss
        if pnls:
            largest_win = max(pnls)
            largest_loss = min(pnls)
            print(f"Largest Win:          ${largest_win:>+10,.2f}")
            print(f"Largest Loss:         ${largest_loss:>+10,.2f}")
        
        # Consecutive wins/losses
        consecutive_stats = self._calculate_consecutive_trades(pnls)
        print(f"Max Consecutive Wins: {consecutive_stats['max_wins']:>10}")
        print(f"Max Consecutive Loss: {consecutive_stats['max_losses']:>10}")
    
    def _calculate_consecutive_trades(self, pnls: List[float]) -> Dict:
        """Calculate consecutive wins/losses"""
        if not pnls:
            return {'max_wins': 0, 'max_losses': 0}
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return {'max_wins': max_wins, 'max_losses': max_losses}
    
    def generate_trade_timeline_chart(self, save_path: str = None):
        """Generate a timeline chart of trades and portfolio value"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("Matplotlib not available for charting")
            return
        
        trades = self.data['trades']
        if not trades:
            print("No trades to chart")
            return
        
        # Calculate portfolio value over time
        starting_balance = self.data['summary']['starting_balance']
        portfolio_values = [starting_balance]
        timestamps = [trades[0]['timestamp']]
        
        current_value = starting_balance
        for trade in trades:
            if trade['type'] == 'CLOSE':
                current_value += trade['pnl']
            elif trade['type'] == 'OPEN':
                current_value -= trade['cost']
            
            portfolio_values.append(current_value)
            timestamps.append(trade['timestamp'])
        
        # Convert timestamps to datetime
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Portfolio value chart
        ax1.plot(dates, portfolio_values, 'b-', linewidth=2)
        ax1.axhline(y=starting_balance, color='gray', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Value Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Trade markers
        for trade in trades:
            trade_date = datetime.fromtimestamp(trade['timestamp'])
            if trade['type'] == 'OPEN':
                ax1.scatter(trade_date, current_value, c='green', marker='^', s=50, alpha=0.7)
            elif trade['type'] == 'CLOSE' and trade['pnl'] > 0:
                ax1.scatter(trade_date, current_value, c='blue', marker='v', s=50, alpha=0.7)
            elif trade['type'] == 'CLOSE' and trade['pnl'] < 0:
                ax1.scatter(trade_date, current_value, c='red', marker='v', s=50, alpha=0.7)
        
        # P&L histogram
        close_trades = [t for t in trades if t['type'] == 'CLOSE']
        if close_trades:
            pnls = [t['pnl'] for t in close_trades]
            ax2.hist(pnls, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Trade P&L ($)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Trade P&L Distribution')
            ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved: {save_path}")
        else:
            plt.show()
    
    def export_to_csv(self, filename: str = None):
        """Export trade data to CSV for further analysis"""
        try:
            import pandas as pd
        except ImportError:
            print("Pandas not available for CSV export")
            return
        
        trades = self.data['trades']
        if not trades:
            print("No trades to export")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Add datetime column
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Reorder columns
        column_order = ['datetime', 'type', 'market', 'side', 'quantity', 'price', 'cost', 'pnl']
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns + [col for col in df.columns if col not in available_columns]]
        
        filename = filename or f"simulation_trades_{int(self.data['trades'][0]['timestamp'])}.csv"
        df.to_csv(filename, index=False)
        print(f"Trade data exported: {filename}")

def main():
    """Main function to analyze simulation results"""
    if len(sys.argv) < 2:
        # Find the most recent simulation log
        log_files = list(Path('.').glob('simulation_log_*.json'))
        if not log_files:
            print("No simulation log files found. Run the simulator first.")
            return
        
        log_file = max(log_files, key=lambda f: f.stat().st_mtime)
        print(f"Using most recent log file: {log_file}")
    else:
        log_file = sys.argv[1]
    
    try:
        analyzer = SimulationAnalyzer(log_file)
        
        # Print summary report
        analyzer.print_summary_report()
        
        # Ask user if they want charts
        try:
            response = input("\nGenerate charts? (y/n): ").lower().strip()
            if response in ('y', 'yes'):
                chart_file = f"simulation_chart_{int(datetime.now().timestamp())}.png"
                analyzer.generate_trade_timeline_chart(chart_file)
        except (KeyboardInterrupt, EOFError):
            pass
        
        # Ask about CSV export
        try:
            response = input("Export to CSV? (y/n): ").lower().strip()
            if response in ('y', 'yes'):
                analyzer.export_to_csv()
        except (KeyboardInterrupt, EOFError):
            pass
            
    except Exception as e:
        print(f"Error analyzing simulation: {e}")

if __name__ == "__main__":
    main()
