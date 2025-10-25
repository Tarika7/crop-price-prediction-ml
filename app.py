"""
Main Streamlit Application - Complete Production Version
Fully integrated ML backend and professional UI
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_backend import (
    load_model, prepare_features, predict_price, generate_recommendation,
    get_best_selling_month, get_model_info, generate_historical_prices,
    predict_all_months, CROP_MODELS
)

from ui_components import (
    render_header, render_sidebar, render_model_info, render_prediction_results,
    render_historical_chart, render_monthly_comparison, render_seasonal_chart,
    render_best_selling_period, render_footer
)

# Page configuration
st.set_page_config(
    page_title="Farm AI - Crop Price Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application flow"""
    
    render_header()
    
    user_inputs = render_sidebar()
    
    if user_inputs and user_inputs['predict_clicked']:
        try:
            with st.spinner('🔄 Loading model and generating predictions...'):
                # Load model
                model = load_model(user_inputs['crop'], user_inputs['state'])
                
                # Prepare features
                features = prepare_features(
                    user_inputs['current_price'],
                    user_inputs['month'],
                    user_inputs['year'],
                    user_inputs['crop']
                )
                
                # Make prediction
                predicted_price = predict_price(model, features)
                
                # Generate recommendation
                recommendation = generate_recommendation(
                    user_inputs['current_price'],
                    predicted_price
                )
                
                # Get model info
                model_info = get_model_info(user_inputs['crop'])
                
                # Get best selling period
                selling_info = get_best_selling_month(user_inputs['crop'])
                
                # Generate historical data
                dates, prices = generate_historical_prices(
                    user_inputs['current_price'],
                    months=12,
                    crop_name=user_inputs['crop'],
                    base_month=user_inputs['month']
                )
                
                # Generate predictions for all months
                predictions_by_month = predict_all_months(
                    model,
                    user_inputs['current_price'],
                    user_inputs['year'],
                    user_inputs['crop']
                )
            
            # Display location context
            st.info(f"📍 **Prediction for:** {user_inputs['crop']} in {user_inputs['district']}, {user_inputs['state']}")
            
            st.success("✅ Prediction Complete!")
            
            # Show model performance
            render_model_info(user_inputs['crop'], model_info)
            
            st.markdown("---")
            
            # Show main prediction
            render_prediction_results(
                user_inputs['current_price'],
                predicted_price,
                recommendation,
                user_inputs['year'],
                user_inputs['month_name']
            )
            
            st.markdown("---")
            
            # Show historical trend
            render_historical_chart(
                dates, prices, predicted_price,
                user_inputs['month_name'],
                user_inputs['year']
            )
            
            st.markdown("---")
            
            # Show monthly comparison
            st.markdown("## 📅 Annual Planning Guide")
            render_monthly_comparison(
                predictions_by_month,
                user_inputs['current_price'],
                user_inputs['year']
            )
            
            st.markdown("---")
            
            # Show seasonal chart
            render_seasonal_chart(
                predictions_by_month,
                user_inputs['current_price'],
                user_inputs['year']
            )
            
            st.markdown("---")
            
            # Show best selling period
            render_best_selling_period(user_inputs['crop'], selling_info)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Farm AI!
        
        This application uses **Machine Learning** to predict agricultural crop prices 
        and help farmers make informed selling decisions.
        
        ### 🎯 Key Features:
        
        - **🎯 Accurate Predictions**: 85-97% accuracy across 8 crops
        - **📊 Visual Insights**: Historical trends and future forecasts
        - **💡 Smart Recommendations**: Data-driven selling advice
        - **📅 Seasonal Analysis**: Full year price comparison
        - **🗺️ Multi-State Support**: Tamil Nadu, Punjab, Haryana
        
        ### 📋 How to Use:
        
        1. **Select Location** - Choose your state and district
        2. **Choose Crop** - Select the crop you want to analyze
        3. **Enter Current Price** - Input today's market price
        4. **Select Date** - Choose year and month for prediction
        5. **Click Predict** - Get comprehensive insights!
        
        **👈 Get started by using the sidebar controls!**
        """)
        
        # Show sample statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Supported States", "3", help="Tamil Nadu, Punjab, Haryana")
        with col2:
            st.metric("Crop Models", "8", help="Rice, Wheat, Vegetables, Pulses, etc.")
        with col3:
            st.metric("Avg Accuracy", "88.5%", help="Average model accuracy")
        with col4:
            st.metric("Data Period", "30+ Years", help="Historical training data")
    
    render_footer()


if __name__ == "__main__":
    main()
