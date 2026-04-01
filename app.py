import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeX Beverage: Price Prediction",
    page_icon="🥤",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background-color: #f8f9fa;
    }

    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.8rem !important;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 0.2rem !important;
    }

    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
    }

    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6c757d;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
    }

    .stSelectbox label, .stNumberInput label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #495057 !important;
    }

    .stSelectbox > div > div {
        border-radius: 8px !important;
        border-color: #dee2e6 !important;
        font-size: 0.9rem !important;
    }

    .stNumberInput > div > div > input {
        border-radius: 8px !important;
        border-color: #dee2e6 !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.03em;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(26,26,46,0.3);
    }

    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
        box-shadow: 0 8px 30px rgba(26,26,46,0.25);
    }

    .result-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.7;
        margin-bottom: 0.5rem;
    }

    .result-value {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-weight: 400;
        margin: 0.3rem 0;
    }

    .result-subtext {
        font-size: 0.85rem;
        opacity: 0.6;
        margin-top: 0.5rem;
    }

    .info-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #0f3460;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }

    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #dee2e6, transparent);
        margin: 1.5rem 0;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helper: Build feature vector ──────────────────────────────────────────────
def preprocess_input(inputs):

    # ── Label encode ordinal columns ──────────────────────────────────────────
    age_group_map = {'18-25': 0, '26-35': 1, '36-45': 2, '46-55': 3, '56-70': 4, '70+': 5}
    income_map    = {'<10L': 0, '10L - 15L': 1, '16L - 25L': 2,
                     '26L - 35L': 3, '> 35L': 4, 'Not Reported': 5}
    health_map    = {'Low': 0, 'Medium': 1, 'High': 2}
    freq_map      = {'0-2 times': 0, '3-4 times': 1, '5-7 times': 2}
    size_map      = {'Small (250 ml)': 0, 'Medium (500 ml)': 1, 'Large (1 L)': 2}

    income    = income_map[inputs['income_levels']]
    freq      = freq_map[inputs['consume_frequency']]
    size      = size_map[inputs['preferable_consumption_size']]
    health    = health_map[inputs['health_concerns']]
    age_group = age_group_map[inputs['age_group']]

    # ── Feature engineering scores ─────────────────────────────────────────────
    freq_score_map   = {'0-2 times': 1, '3-4 times': 2, '5-7 times': 3}
    aware_score_map  = {'0 to 1': 1, '2 to 4': 2, 'above 4': 3}
    zone_score_map   = {'Urban': 3, 'Metro': 4, 'Rural': 1, 'Semi-Urban': 2}
    income_score_map = {'<10L': 1, '10L - 15L': 2, '16L - 25L': 3,
                        '26L - 35L': 4, '> 35L': 5, 'Not Reported': 0}

    freq_score      = freq_score_map[inputs['consume_frequency']]
    awareness_score = aware_score_map[inputs['awareness_of_other_brands']]
    cf_ab_score     = round(freq_score / (awareness_score + freq_score), 2)

    zas_score = zone_score_map[inputs['zone']] * income_score_map[inputs['income_levels']]

    cond1 = inputs['current_brand'] != 'Established'
    cond2 = inputs['reasons_for_choosing_brands'] in ['Price', 'Quality']
    bsi   = 1 if (cond1 and cond2) else 0

    # ── OHE columns exactly matching X_train ──────────────────────────────────
    # gender: F / M (not Female/Male)
    gender_val = 'F' if inputs['gender'] == 'Female' else 'M'

    # situation mapping to exact column names
    situation_map = {
        'Active (eg. Sports, gym)' : 'typical_consumption_situations_Active__eg__Sports__gym',
        'Casual (At home)'         : 'typical_consumption_situations_Casual__eg__At_home',
        'Social (Parties)'         : 'typical_consumption_situations_Social__eg__Parties'
    }

    # ── Build full feature dict in EXACT column order ──────────────────────────
    row = {
        # Label encoded
        'income_levels'              : income,
        'consume_frequency_weekly'   : freq,       # no trailing underscore
        'preferable_consumption_size': size,
        'health_concerns'            : health,
        'age_group'                  : age_group,
        # Engineered
        'cf_ab_score'                : cf_ab_score,
        'zas_score'                  : zas_score,
        'bsi'                        : bsi,
        # Gender OHE — F/M not Female/Male
        'gender_F'                   : 1 if gender_val == 'F' else 0,
        'gender_M'                   : 1 if gender_val == 'M' else 0,
        # Zone OHE
        'zone_Metro'                 : 1 if inputs['zone'] == 'Metro'      else 0,
        'zone_Rural'                 : 1 if inputs['zone'] == 'Rural'      else 0,
        'zone_Semi_Urban'            : 1 if inputs['zone'] == 'Semi-Urban' else 0,
        'zone_Urban'                 : 1 if inputs['zone'] == 'Urban'      else 0,
        # Occupation OHE
        'occupation_Entrepreneur'         : 1 if inputs['occupation'] == 'Entrepreneur'        else 0,
        'occupation_Retired'              : 1 if inputs['occupation'] == 'Retired'             else 0,
        'occupation_Student'              : 1 if inputs['occupation'] == 'Student'             else 0,
        'occupation_Working_Professional' : 1 if inputs['occupation'] == 'Working Professional' else 0,
        # Brand OHE
        'current_brand_Established' : 1 if inputs['current_brand'] == 'Established' else 0,
        'current_brand_Newcomer'    : 1 if inputs['current_brand'] == 'Newcomer'    else 0,
        # Awareness OHE
        'awareness_of_other_brands_0_to_1'  : 1 if inputs['awareness_of_other_brands'] == '0 to 1'  else 0,
        'awareness_of_other_brands_2_to_4'  : 1 if inputs['awareness_of_other_brands'] == '2 to 4'  else 0,
        'awareness_of_other_brands_above_4' : 1 if inputs['awareness_of_other_brands'] == 'above 4' else 0,
        # Reasons OHE
        'reasons_for_choosing_brands_Availability'      : 1 if inputs['reasons_for_choosing_brands'] == 'Availability'      else 0,
        'reasons_for_choosing_brands_Brand_Reputation'  : 1 if inputs['reasons_for_choosing_brands'] == 'Brand Reputation'  else 0,
        'reasons_for_choosing_brands_Price'             : 1 if inputs['reasons_for_choosing_brands'] == 'Price'             else 0,
        'reasons_for_choosing_brands_Quality'           : 1 if inputs['reasons_for_choosing_brands'] == 'Quality'           else 0,
        # Flavor OHE
        'flavor_preference_Exotic'      : 1 if inputs['flavor_preference'] == 'Exotic'      else 0,
        'flavor_preference_Traditional' : 1 if inputs['flavor_preference'] == 'Traditional' else 0,
        # Channel OHE
        'purchase_channel_Online'       : 1 if inputs['purchase_channel'] == 'Online'       else 0,
        'purchase_channel_Retail_Store' : 1 if inputs['purchase_channel'] == 'Retail Store' else 0,
        # Packaging OHE
        'packaging_preference_Eco_Friendly' : 1 if inputs['packaging_preference'] == 'Eco-Friendly' else 0,
        'packaging_preference_Premium'      : 1 if inputs['packaging_preference'] == 'Premium'      else 0,
        'packaging_preference_Simple'       : 1 if inputs['packaging_preference'] == 'Simple'       else 0,
        # Situation OHE — exact column names from model
        'typical_consumption_situations_Active__eg__Sports__gym' : 1 if inputs['typical_consumption_situations'] == 'Active (eg. Sports, gym)' else 0,
        'typical_consumption_situations_Casual__eg__At_home'     : 1 if inputs['typical_consumption_situations'] == 'Casual (At home)'         else 0,
        'typical_consumption_situations_Social__eg__Parties'     : 1 if inputs['typical_consumption_situations'] == 'Social (Parties)'         else 0,
    }

    return pd.DataFrame([row])


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1> CodeX Beverage: Price Prediction</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the preferred price range of a customer using our best-performing LightGBM model</p>',
            unsafe_allow_html=True)

# ── Row 1: Demographics ───────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title"> Demographics</div>',
            unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    age        = st.number_input("Age", min_value=18, max_value=70, value=28)
with c2:
    gender     = st.selectbox("Gender", ["Male", "Female"])
with c3:
    zone       = st.selectbox("Zone", ["Urban", "Metro", "Rural", "Semi-Urban"])
with c4:
    occupation = st.selectbox("Occupation", ["Working Professional", "Student",
                                              "Entrepreneur", "Retired"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Row 2: Purchase Behaviour ─────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title"> Purchase Behaviour</div>',
            unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    income         = st.selectbox("Income Level (In L)",
                                   ["<10L", "10L - 15L", "16L - 25L",
                                    "26L - 35L", "> 35L", "Not Reported"])
with c2:
    consume_freq   = st.selectbox("Consume Frequency (Weekly)",
                                   ["0-2 times", "3-4 times", "5-7 times"])
with c3:
    current_brand  = st.selectbox("Current Brand", ["Newcomer", "Established"])
with c4:
    pref_size      = st.selectbox("Preferable Consumption Size",
                                   ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Row 3: Preferences ────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title"> Preferences & Awareness</div>',
            unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    awareness      = st.selectbox("Awareness of Other Brands",
                                   ["0 to 1", "2 to 4", "above 4"])
with c2:
    reasons        = st.selectbox("Reasons for Choosing Brands",
                                   ["Price", "Quality", "Availability", "Brand Reputation"])
with c3:
    flavor         = st.selectbox("Flavor Preference", ["Traditional", "Exotic"])
with c4:
    channel        = st.selectbox("Purchase Channel", ["Online", "Retail Store"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Row 4: Other ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title"> Packaging & Lifestyle</div>',
            unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    packaging      = st.selectbox("Packaging Preference",
                                   ["Simple", "Premium", "Eco-Friendly"])
with c2:
    health         = st.selectbox("Health Concerns", ["Low", "Medium", "High"])
with c3:
    situation      = st.selectbox("Typical Consumption Situations",
                                   ["Active (eg. Sports, gym)",
                                    "Social (Parties)",
                                    "Casual (At home)"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Age Group helper ──────────────────────────────────────────────────────────
def get_age_group(age):
    if age <= 25:   return '18-25'
    elif age <= 35: return '26-35'
    elif age <= 45: return '36-45'
    elif age <= 55: return '46-55'
    elif age <= 70: return '56-70'
    else:           return '70+'

# ── Predict Button ────────────────────────────────────────────────────────────
col_btn, col_result = st.columns([1, 2])

with col_btn:
    predict_btn = st.button(" Calculate Price Range")

with col_result:
    if predict_btn:
        inputs = {
            'age'                          : age,
            'age_group'                    : get_age_group(age),
            'gender'                       : gender,
            'zone'                         : zone,
            'occupation'                   : occupation,
            'income_levels'                : income,
            'consume_frequency'            : consume_freq,
            'current_brand'                : current_brand,
            'preferable_consumption_size'  : pref_size,
            'awareness_of_other_brands'    : awareness,
            'reasons_for_choosing_brands'  : reasons,
            'flavor_preference'            : flavor,
            'purchase_channel'             : channel,
            'packaging_preference'         : packaging,
            'health_concerns'              : health,
            'typical_consumption_situations': situation
        }

        try:
            # Load model
            with open('lgbm_model.pkl', 'rb') as f:
                model = pickle.load(f)

            X_input = preprocess_input(inputs)
            pred    = model.predict(X_input)[0]

            price_labels = {0: '₹100 - ₹150', 1: '₹150 - ₹200',
                            2: '₹200 - ₹250', 3: '₹50 - ₹100'}
            price_range  = price_labels.get(pred, str(pred))

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Price Range</div>
                <div class="result-value">{price_range}</div>
                <div class="result-subtext">Based on customer profile analysis</div>
                <div class="info-badge"> LightGBM · 92.16% Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("⚠️ Model file 'lgbm_model.pkl' not found. Please save your trained model first.")
            st.code("""
# Run this in your Jupyter notebook to save the model:
import pickle
with open('lgbm_model.pkl', 'wb') as f:
    pickle.dump(models['LGBMClassifier'], f)
print("✅ Model saved!")
            """)