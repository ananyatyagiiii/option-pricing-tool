
Option pricing tool · PY
import numpy as np
import matplotlib
 
matplotlib.use("Agg")  # Non-interactive backend (safe for CI / headless runs)
import matplotlib.pyplot as plt
from scipy.stats import norm
 
 
# ============================================================================
# PRICING FUNCTIONS
# ============================================================================
 
def black_scholes_put(S0, K, T, r, sigma):
    """Black-Scholes analytical price for a European PUT option.
 
    S0    = current stock price
    K     = strike price
    T     = time to expiration (in years)
    r     = risk-free rate (e.g. 0.05 for 5%)
    sigma = annualized volatility
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
 
    # Key difference from the call: use NEGATIVE d1 and d2
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
 
    return put_price
 
 
def black_scholes_call(S0, K, T, r, sigma):
    """Black-Scholes analytical price for a European CALL option.
 
    S0    = current stock price
    K     = strike price
    T     = time to expiration (in years)
    r     = risk-free rate (e.g. 0.05 for 5%)
    sigma = annualized volatility
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
 
    # norm.cdf() answers "what's the probability?"
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
 
    call_price = S0 * N_d1 - K * np.exp(-r * T) * N_d2
 
    return call_price
 
 
def calculate_option_price(initial_price, strike_price, volatility, days,
                           simulations, r, T, option_type='call'):
    """Price a European option by Monte Carlo simulation.
 
    Returns (option_price, last_simulated_path).
    """
    # Convert annual volatility to daily volatility
    daily_volatility = volatility / np.sqrt(252)  # 252 trading days per year
 
    payoffs = []
    prices = [initial_price]
    for sim in range(simulations):
        price = initial_price
        prices = [price]
        for day in range(days):
            daily_return_mean = r / 252  # Convert annual rate to daily
            daily_return_std = daily_volatility
            random_return = np.random.normal(daily_return_mean, daily_return_std)
            price *= (1 + random_return)
            prices.append(price)
        final_price = prices[-1]
        if option_type == 'call':
            payoff = max(final_price - strike_price, 0)
        elif option_type == 'put':
            payoff = max(strike_price - final_price, 0)  # Note: strike FIRST
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        payoffs.append(payoff)
 
    option_price = np.mean(payoffs) * np.exp(-r * T)
    return option_price, prices
 
 
# ============================================================================
# INTERACTIVE SCRIPT
# ============================================================================
 
def main():
    initial_price = float(input("Enter initial stock price: "))
    strike_price = float(input("Enter strike price: "))
    days = int(input("Enter days to expiration: "))
    simulations_input = input("Enter number of simulations (default 100000): ")
    simulations = int(simulations_input) if simulations_input else 100000
 
    # Preset volatility scenarios
    vol_low = 0.05      # Calm market
    vol_medium = 0.15   # Normal market
    vol_high = 0.30     # Volatile market
 
    T = days / 365
    r = 0.05
 
    result1 = calculate_option_price(initial_price, strike_price, vol_low, days, simulations, r, T)
    result2 = calculate_option_price(initial_price, strike_price, vol_medium, days, simulations, r, T)
    result3 = calculate_option_price(initial_price, strike_price, vol_high, days, simulations, r, T)
 
    # Black-Scholes comparison
    bs_vol_low = black_scholes_call(initial_price, strike_price, T, r, vol_low)
    bs_vol_medium = black_scholes_call(initial_price, strike_price, T, r, vol_medium)
    bs_vol_high = black_scholes_call(initial_price, strike_price, T, r, vol_high)
 
    print(f"Fair option price (strike_price, vol_low): ${result1[0]:.2f}")
    print(f"Fair option price (strike_price, vol_medium): ${result2[0]:.2f}")
    print(f"Fair option price (strike_price, vol_high): ${result3[0]:.2f}")
    print("\n--- BLACK-SCHOLES COMPARISON ---")
    print(f"Vol 5%:  MC=${result1[0]:.2f}  vs  BS=${bs_vol_low:.2f}")
    print(f"Vol 15%: MC=${result2[0]:.2f}  vs  BS=${bs_vol_medium:.2f}")
    print(f"Vol 30%: MC=${result3[0]:.2f}  vs  BS=${bs_vol_high:.2f}")
 
    # Calculate PUT prices
    put_result1 = calculate_option_price(initial_price, strike_price, vol_low, days, simulations, r, T, option_type='put')
    put_result2 = calculate_option_price(initial_price, strike_price, vol_medium, days, simulations, r, T, option_type='put')
    put_result3 = calculate_option_price(initial_price, strike_price, vol_high, days, simulations, r, T, option_type='put')
 
    # Black-Scholes PUT prices
    bs_put_vol_low = black_scholes_put(initial_price, strike_price, T, r, vol_low)
    bs_put_vol_medium = black_scholes_put(initial_price, strike_price, T, r, vol_medium)
    bs_put_vol_high = black_scholes_put(initial_price, strike_price, T, r, vol_high)
 
    print("\n--- PUT OPTIONS ---")
    print(f"Vol 5%:  MC=${put_result1[0]:.2f}  vs  BS=${bs_put_vol_low:.2f}")
    print(f"Vol 15%: MC=${put_result2[0]:.2f}  vs  BS=${bs_put_vol_medium:.2f}")
    print(f"Vol 30%: MC=${put_result3[0]:.2f}  vs  BS=${bs_put_vol_high:.2f}")
 
    # ========================================================================
    # PROFESSIONAL VISUALIZATIONS
    # ========================================================================
 
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
 
    # Plot 1: Stock Price Paths
    ax1 = fig.add_subplot(gs[0, 0])
    times = np.linspace(0, T, len(result1[1]))
    for result, vol, label in [(result1, vol_low, '5%'),
                               (result2, vol_medium, '15%'),
                               (result3, vol_high, '30%')]:
        ax1.plot(times, result[1], label=f'Vol: {label}', alpha=0.7, linewidth=2)
    ax1.axhline(strike_price, color='red', linestyle='--', linewidth=2, label='Strike')
    ax1.axhline(initial_price, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Stock Price ($)')
    ax1.set_title('Sample Stock Price Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    # Plot 2: Call Option Prices - MC vs Black-Scholes
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ['Vol 5%', 'Vol 15%', 'Vol 30%']
    mc_calls = [result1[0], result2[0], result3[0]]
    bs_calls = [bs_vol_low, bs_vol_medium, bs_vol_high]
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width / 2, mc_calls, width, label='Monte Carlo', alpha=0.8)
    ax2.bar(x + width / 2, bs_calls, width, label='Black-Scholes', alpha=0.8)
    ax2.set_ylabel('Option Price ($)')
    ax2.set_title('CALL Option Prices: MC vs Analytical')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
 
    # Plot 3: Put Option Prices - MC vs Black-Scholes
    ax3 = fig.add_subplot(gs[1, 0])
    mc_puts = [put_result1[0], put_result2[0], put_result3[0]]
    bs_puts = [bs_put_vol_low, bs_put_vol_medium, bs_put_vol_high]
    ax3.bar(x - width / 2, mc_puts, width, label='Monte Carlo', alpha=0.8, color='coral')
    ax3.bar(x + width / 2, bs_puts, width, label='Black-Scholes', alpha=0.8, color='lightcoral')
    ax3.set_ylabel('Option Price ($)')
    ax3.set_title('PUT Option Prices: MC vs Analytical')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
 
    # Plot 4: Payoff Diagrams
    ax4 = fig.add_subplot(gs[1, 1])
    stock_range = np.linspace(initial_price * 0.7, initial_price * 1.3, 100)
    call_payoff = np.maximum(stock_range - strike_price, 0)
    put_payoff = np.maximum(strike_price - stock_range, 0)
    ax4.plot(stock_range, call_payoff, linewidth=2.5, label='Call Payoff', color='green')
    ax4.plot(stock_range, put_payoff, linewidth=2.5, label='Put Payoff', color='red')
    ax4.axvline(initial_price, color='black', linestyle='--', linewidth=2, label=f'Current: ${initial_price}')
    ax4.axvline(strike_price, color='blue', linestyle='--', linewidth=2, label=f'Strike: ${strike_price}')
    ax4.fill_between(stock_range, 0, call_payoff, alpha=0.2, color='green')
    ax4.fill_between(stock_range, 0, put_payoff, alpha=0.2, color='red')
    ax4.set_xlabel('Stock Price at Expiration ($)')
    ax4.set_ylabel('Payoff ($)')
    ax4.set_title('Option Payoff Diagrams')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
 
    # Plot 5: Distribution of Final Prices
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist([result1[1], result2[1], result3[1]],
             bins=30, alpha=0.6, label=['Vol 5%', 'Vol 15%', 'Vol 30%'], edgecolor='black')
    ax5.axvline(strike_price, color='red', linestyle='--', linewidth=2, label='Strike')
    ax5.axvline(initial_price, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Current')
    ax5.set_xlabel('Stock Price ($)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Simulated Prices')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
 
    # Plot 6: Summary Table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_data = [
        ['Volatility', 'Call MC', 'Call BS', 'Put MC', 'Put BS'],
        ['5%', f'${result1[0]:.2f}', f'${bs_vol_low:.2f}', f'${put_result1[0]:.2f}', f'${bs_put_vol_low:.2f}'],
        ['15%', f'${result2[0]:.2f}', f'${bs_vol_medium:.2f}', f'${put_result2[0]:.2f}', f'${bs_put_vol_medium:.2f}'],
        ['30%', f'${result3[0]:.2f}', f'${bs_vol_high:.2f}', f'${put_result3[0]:.2f}', f'${bs_put_vol_high:.2f}'],
    ]
    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    ax6.set_title('Price Summary', fontsize=12, fontweight='bold', pad=0)
 
    plt.savefig('option_pricing_dashboard.png', dpi=100, bbox_inches='tight')
    print("\n✓ Professional dashboard saved to option_pricing_dashboard.png")
 
 
if __name__ == "__main__":
    main()
 


