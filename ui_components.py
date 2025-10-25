"""
UI Components Module - Complete Production Version
Professional UI with all visualizations and interactivity
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PRIMARY_COLOR = "#1f8b4c"
SECONDARY_COLOR = "#f39c12"


def render_header():
    """Render professional application header"""
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1.5rem 0;
            background: linear-gradient(90deg, #1f8b4c 0%, #27ae60 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="main-header">
            <h1>🌾 Farm AI - Crop Price Predictor</h1>
            <p>Powered by Machine Learning & Real Agricultural Data</p>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with all input controls"""
    from ml_backend import get_available_states, get_state_crops, get_state_districts
    
    st.sidebar.header("📊 Prediction Parameters")
    
    # State selection
    st.sidebar.markdown("### 📍 Location")
    available_states = get_available_states()
    selected_state = st.sidebar.selectbox(
        "Select State/Region",
        available_states,
        help="Choose your state or region"
    )
    
    # District selection
    districts = get_state_districts(selected_state)
    selected_district = st.sidebar.selectbox(
        "Select District",
        districts,
        help="Choose your district for localized insights"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌾 Crop Selection")
    
    # Crop selection
    available_crops = get_state_crops(selected_state)
    if not available_crops:
        st.sidebar.error(f"No crops available for {selected_state}")
        return None
    
    selected_crop = st.sidebar.selectbox(
        "Select Crop",
        available_crops,
        help="Choose the crop for price prediction"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Market Data")
    
    # Current price
    current_price = st.sidebar.number_input(
        "Current Market Price (₹/quintal)",
        min_value=100.0,
        max_value=20000.0,
        value=2500.0,
        step=100.0,
        help="Enter the current market price"
    )
    
    # Year selection
    current_year = pd.Timestamp.now().year
    selected_year = st.sidebar.selectbox(
        "Select Year",
        range(current_year, current_year + 3),
        help="Choose the year for prediction"
    )
    
    # Month selection
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    selected_month = st.sidebar.selectbox(
        "Select Month",
        months,
        index=pd.Timestamp.now().month - 1,
        help="Choose the month for prediction"
    )
    month_num = months.index(selected_month) + 1
    
    # Predict button
    predict_button = st.sidebar.button(
        "🔮 Predict Price",
        use_container_width=True,
        type="primary"
    )
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Selected Parameters:**
    - **State:** {selected_state}
    - **District:** {selected_district}
    - **Crop:** {selected_crop}
    - **Year:** {selected_year}
    - **Month:** {selected_month}
    
    Click **Predict Price** to get insights!
    """)
    
    return {
        'state': selected_state,
        'district': selected_district,
        'crop': selected_crop,
        'current_price': current_price,
        'month': month_num,
        'year': selected_year,
        'month_name': selected_month,
        'predict_clicked': predict_button
    }


def render_model_info(crop_name, model_info):
    """Display model performance metrics"""
    st.markdown("### 🎯 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", f"{model_info['accuracy']:.1f}%", help="Prediction accuracy on test data")
    
    with col2:
        st.metric("Model Type", model_info['model_type'], help="Machine learning algorithm")
    
    with col3:
        st.metric("Training Period", model_info['training_period'], help="Historical data period")
    
    with col4:
        st.metric("Features Used", model_info['features_used'], help="Input features count")


def render_prediction_results(current_price, predicted_price, recommendation, year, month_name):
    """Display prediction results"""
    st.markdown(f"### 📈 Prediction Results for {month_name} {year}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"₹{current_price:,.2f}", help="Today's market price")
    
    with col2:
        st.metric(
            "Predicted Price",
            f"₹{predicted_price:,.2f}",
            delta=f"{recommendation['change_pct']:+.1f}%",
            help=f"Predicted price for {month_name} {year}"
        )
    
    with col3:
        st.metric("Expected Change", f"₹{recommendation['change_amount']:+,.2f}", help="Absolute price change")
    
    st.markdown("---")
    st.markdown("### 💡 Recommendation")
    
    if recommendation['status'] == "WAIT":
        st.success(f"{recommendation['icon']} **{recommendation['message']}**")
    elif recommendation['status'] == "SELL":
        st.error(f"{recommendation['icon']} **{recommendation['message']}**")
    else:
        st.info(f"{recommendation['icon']} **{recommendation['message']}**")
    
    st.write(recommendation['advice'])


def render_historical_chart(dates, prices, predicted_price, month_name, year):
    """Render price trend visualization"""
    st.markdown("### 📊 Historical Price Trend & Forecast")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(dates, prices, marker='o', linewidth=2.5, label='Historical Prices', color='#3498db', markersize=6)
    
    last_date = dates[-1]
    next_month = pd.date_range(start=last_date, periods=2, freq='M')[1]
    
    ax.plot([last_date, next_month], [prices[-1], predicted_price],
            marker='o', linewidth=2.5, linestyle='--', label=f'Prediction ({month_name} {year})', 
            color='#e74c3c', markersize=8)
    
    ax.axhline(y=predicted_price, color='red', linestyle=':', alpha=0.5, label='Predicted Price Level')
    
    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax.set_ylabel('Price (₹/quintal)', fontweight='bold', fontsize=12)
    ax.set_title('12-Month Historical Trend & Future Prediction', fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)


def render_monthly_comparison(predictions_by_month, current_price, year):
    """Show predictions for all 12 months"""
    st.markdown(f"### 📅 Full Year Price Forecast - {year}")
    st.markdown("Compare predicted prices across all months to plan your selling strategy:")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    best_month_idx = predictions_by_month.index(max(predictions_by_month))
    worst_month_idx = predictions_by_month.index(min(predictions_by_month))
    
    df = pd.DataFrame({
        'Month': months,
        'Predicted Price (₹)': [f"₹{p:,.2f}" for p in predictions_by_month],
        'Change from Current (%)': [f"{((p - current_price) / current_price * 100):+.1f}%" for p in predictions_by_month],
        'Price Level': ['🟢 Best' if i == best_month_idx else '🔴 Worst' if i == worst_month_idx else '⚪' for i in range(12)]
    })
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🌟 **Best Month:** {months[best_month_idx]} (₹{predictions_by_month[best_month_idx]:,.2f})")
    with col2:
        st.error(f"⚠️ **Worst Month:** {months[worst_month_idx]} (₹{predictions_by_month[worst_month_idx]:,.2f})")
    with col3:
        price_range = max(predictions_by_month) - min(predictions_by_month)
        st.info(f"📊 **Price Range:** ₹{price_range:,.2f}")


def render_seasonal_chart(predictions_by_month, current_price, year):
    """Show seasonal price variation chart"""
    st.markdown(f"### 📊 Seasonal Price Variation - {year}")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    ax.plot(months, predictions_by_month, marker='o', linewidth=3, color='#2ecc71', label='Predicted Prices', markersize=8)
    ax.axhline(y=current_price, color='blue', linestyle='--', alpha=0.6, label='Current Price', linewidth=2)
    
    best_idx = predictions_by_month.index(max(predictions_by_month))
    worst_idx = predictions_by_month.index(min(predictions_by_month))
    
    ax.scatter([best_idx], [predictions_by_month[best_idx]], color='green', s=300, zorder=5, label='Best Month', marker='*')
    ax.scatter([worst_idx], [predictions_by_month[worst_idx]], color='red', s=300, zorder=5, label='Worst Month', marker='v')
    
    ax.fill_between(range(12), predictions_by_month, current_price, alpha=0.2, color='green', 
                      where=[p > current_price for p in predictions_by_month])
    ax.fill_between(range(12), predictions_by_month, current_price, alpha=0.2, color='red',
                      where=[p < current_price for p in predictions_by_month])
    
    ax.set_xlabel('Month', fontweight='bold', fontsize=13)
    ax.set_ylabel('Price (₹/quintal)', fontweight='bold', fontsize=13)
    ax.set_title(f'Seasonal Price Forecast for {year}', fontweight='bold', fontsize=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    st.pyplot(fig)


def render_best_selling_period(crop_name, selling_info):
    """Display optimal selling period"""
    st.markdown("### 📅 Recommended Selling Period")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Optimal Period:**")
        st.markdown(f"### {selling_info['months']}")
    
    with col2:
        st.markdown("**Reason:**")
        st.write(selling_info['reason'])


def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p style='margin: 0; font-size: 1.1rem;'><b>🌾 Farm AI - Crop Price Predictor</b></p>
            <p style='margin: 0.5rem 0;'>Developed with ❤️ for Indian Farmers</p>
            <p style='margin: 0; font-size: 0.9rem;'>Powered by Random Forest ML | 30+ Years Agricultural Data | 85-97% Accuracy</p>
        </div>
    """, unsafe_allow_html=True)
