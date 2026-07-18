import joblib
import streamlit as st
import pandas as pd
import numpy as np
import shap        
import warnings    

# --- IMPORT FROM CUSTOM MODULES ---
from modules.ml_engine import load_engine, get_flat_shap
from modules.visual_ui import inject_custom_css, draw_process_graph, render_custom_metric
from modules.agentic_ai import generate_risk_narrative, generate_detailed_business_report, run_autonomous_agent

# Now warnings will work
warnings.filterwarnings('ignore')

# --- HELPER FUNCTION (Code Cleanup) ---
def render_risk_badge(prob):
    """Consolidates the HTML rendering for risk scores to prevent duplicated code."""
    if prob > 0.5:
        st.markdown(f"<h2 style='color:#f87171; margin-bottom:0; text-shadow: 0 0 20px rgba(248, 113, 113, 0.4);'>MANUAL REVIEW</h2>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size: 1.2rem; font-weight: 600; color: #fca5a5;'>Risk Score: {prob*100:.1f}%</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:#4ade80; margin-bottom:0; text-shadow: 0 0 20px rgba(74, 222, 128, 0.4);'>LOW RISK</h2>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size: 1.2rem; font-weight: 600; color: #86efac;'>Risk Score: {prob*100:.1f}%</span>", unsafe_allow_html=True)

# 1. ENHANCED CACHE FUNCTION
@st.cache_resource
def load_all_resources():
    """
    Caches everything once. We use the existing load_engine() 
    logic but wrap it in Streamlit's cache to prevent 5-10s reloads.
    """
    # Using your existing module to fetch the components
    df, X, model, explainer, encoders, acc = load_engine()
    return df, X, model, explainer, encoders, acc

# 2. INITIALIZE ONCE
# This replaces the double-loading logic
df, X, model, explainer, encoders, acc = load_all_resources()

# --- PAGE SETUP ---
st.set_page_config(page_title="Agentic O2C Process Miner", layout="wide", initial_sidebar_state="expanded")
inject_custom_css()

if df is not None:
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9523/9523671.png", width=50)
        st.markdown("<h3 style='margin-top:0;'>Admin Console</h3>", unsafe_allow_html=True)
        st.markdown("---")
        st.success("● Predictive ML Engine Online")
        st.success("● Causal SHAP Visuals Active")
        st.success("● Prescriptive AI Agent Online") 
        st.markdown("---")
        
        # --- UI FIX: Use real academic metrics ---
        acc_val = model.test_metrics['accuracy'] if hasattr(model, 'test_metrics') else acc
        render_custom_metric("Test Accuracy", f"{acc_val*100:.1f}%", "Out-of-sample Validation")
        st.caption("Random Forest • 50 Estimators • SHAP")

    # --- MAIN PAGE ---
    st.title(" Agentic Decision Intelligence: O2C")
    st.markdown("_Leveraging Predictive ML, SHAP, and Prescriptive AI Agents_")

    tab1, tab2, tab3, tab4 = st.tabs(["🔎 Case Inspector", "📈 Process Analytics", "🔮 Risk Simulator", "🤖 Agentic Business Report"])

    # ==========================================
    # === TAB 1: CASE INSPECTOR (NO AI AGENT) ==
    # ==========================================
    with tab1:
        st.subheader("Case Grouping & Selection")
       # 1. Cohort Selection
        selected_path = st.selectbox("Select Process Sequence Group:", sorted(df['Process_Path_Group'].value_counts().index))
        group_df = df[df['Process_Path_Group'] == selected_path].drop_duplicates(subset=['case_id'])
        
        # --- UI FIX: 3 even columns to perfectly center the KPI boxes ---
        s1, s2, s3 = st.columns(3)
        with s1: 
            render_custom_metric("Cases in Group", f"{len(group_df)}")
        with s2: 
            render_custom_metric("Avg Order Value", f"${group_df['order_value'].mean():,.0f}")
        with s3: 
            render_custom_metric("Avg Process Time", f"{group_df['processing_days'].mean():.1f} d")
            
        st.markdown("---")
            
        st.markdown(f"#### 📋 Case List: {selected_path}")
        preview_df = group_df[['case_id', 'order_value', 'processing_days', 'has_manual_review']].copy()
        preview_df['Status'] = preview_df['has_manual_review'].apply(lambda x: "⛔ Review" if (x==1 or str(x).upper()=='TRUE') else "✅ Low Risk")
        
        selection = st.dataframe(preview_df.head(100), use_container_width=True, hide_index=True, height=250, on_select="rerun", selection_mode="single-row")
        
        if len(selection.selection.rows) > 0:
            selected_case_id = preview_df.iloc[selection.selection.rows[0]]['case_id']
            row_idx = df[df['case_id'] == selected_case_id].index[0]
            case_data = df.iloc[row_idx]
            X_case = X.iloc[[row_idx]]
            prob = model.predict_proba(X_case)[0][1]
            is_actual_risk = True if case_data['has_manual_review'] == 1 else False
            
            st.markdown("---")
            st.subheader(f"Deep Dive Analysis: {selected_case_id}")
            col_proc, col_reason = st.columns([1.8, 1.2])
            
            with col_proc:
                st.markdown("#### 📍 Process Sequence Map")
                st.graphviz_chart(draw_process_graph(is_actual_risk), use_container_width=True)
                
                st.markdown("#### 📦 Shipment Metadata")
                m1, m2, m3, m4 = st.columns(4)
                with m1: render_custom_metric("Value", f"${case_data.get('order_value', 0):,.0f}")
                with m2: render_custom_metric("Weight", f"{case_data.get('package_weight_kg', 0)}kg")
                with m3: render_custom_metric("Staff", case_data.get('staff_training_level', 'N/A'))
                with m4: render_custom_metric("Category", case_data.get('product_type', 'N/A'))
            
            with col_reason:
                st.markdown("#### 🧠 Predictive Diagnosis")
                
                # --- UI FIX: Use the new helper function ---
                render_risk_badge(prob)
                
                st.markdown("---")
                st.markdown("#### 🏆 Causal SHAP Drivers")
                vals = get_flat_shap(explainer, X_case, len(X.columns))
                impact_df = pd.DataFrame({'Feature': X.columns, 'Contribution': vals}).sort_values(by='Contribution', ascending=False)
                
                for i, row in impact_df.head(4).iterrows():
                    if row['Contribution'] > 0:
                        feat_name = row['Feature'].title().replace('_', ' ')
                        contrib = row['Contribution']

                        num_icons = 5 if contrib > 0.25 else (4 if contrib > 0.20 else (3 if contrib > 0.15 else (2 if contrib > 0.10 else 1)))
                        st.markdown(f"{'🔥' * num_icons} **{feat_name}** (+{contrib*100:.1f}%)")
        else:
             st.info("👆 Click on any row to reveal the Case Inspector Deep Dive.")

    # ==========================================
    # === TAB 2: ANALYTICS (PRESERVED) =========
    # ==========================================
    with tab2:
        st.subheader("Process Variants & Bottlenecks")
        variant_stats = df.groupby('Variant_Name').agg({
            'case_id': 'nunique', 'processing_days': 'mean', 'order_value': 'mean', 'has_manual_review': lambda x: np.mean(x) * 100 
        }).reset_index()
        variant_stats.columns = ['Variant', 'Case Volume', 'Avg Time (Days)', 'Avg Order Value ($)', 'Manual Review %']
        variant_stats['Avg Time (Days)'] = variant_stats['Avg Time (Days)'].round(1)
        variant_stats['Avg Order Value ($)'] = variant_stats['Avg Order Value ($)'].round(0)
        variant_stats['Manual Review %'] = variant_stats['Manual Review %'].round(1)
        
        st.dataframe(variant_stats.sort_values(by='Manual Review %', ascending=False).style.background_gradient(subset=['Avg Time (Days)'], cmap='RdPu'), use_container_width=True)
        
        col_select, col_kpi = st.columns([1, 3])
        with col_select:
            target_variant = st.selectbox("Select Variant to Analyze:", variant_stats['Variant'].unique())
        variant_df = df[df['Variant_Name'] == target_variant]
        
        with col_kpi:
            k1, k2, k3, k4 = st.columns(4)
            with k1: render_custom_metric("Throughput Time", f"{variant_df['processing_days'].mean():.1f} Days", "vs Avg")
            with k2: render_custom_metric("Manual Review %", f"{variant_df['has_manual_review'].mean()*100:.1f}%", "Review Freq", "neg")
            with k3: render_custom_metric("Avg Order Value", f"${variant_df['order_value'].mean():,.0f}", "Financial")
            with k4: render_custom_metric("Total Volume", f"{variant_df['case_id'].nunique()}", "Cases")

        st.markdown("---")
        st.markdown("#### 📋 Causal Impact Values")
        if len(variant_df) > 50:
            X_variant = X.loc[variant_df.index].head(100)
            shap_values_var = explainer.shap_values(X_variant)
            vals = shap_values_var[1] if isinstance(shap_values_var, list) else shap_values_var
            if len(vals.shape) == 3: vals = vals[:, :, 1]
            
            shap_df = pd.DataFrame({'Feature': X.columns, 'Impact': np.mean(vals, axis=0)})
            shap_df = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(5)
            
            if not shap_df.empty:
                cols = st.columns(len(shap_df))
                for idx, (index, row) in enumerate(shap_df.iterrows()):
                    with cols[idx]:
                        st.markdown(f"""
                            <div style="text-align: left; background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px;">
                                <p style="font-size: 0.8rem; font-weight: 600; color: #94a3b8; margin-bottom: 4px;">{row['Feature'].replace('_', ' ').title()}</p>
                                <p style="font-size: 1.2rem; font-weight: 800; color: #38bdf8; margin: 0;">{row['Impact']:.4f}</p>
                            </div>
                        """, unsafe_allow_html=True)
            else: st.info("No consistent risk-increasing drivers found for this variant.")
        else: st.warning("Insufficient data for Causal Impact analysis.")

    # ==========================================
    # === TAB 3: RISK SIMULATOR (ENRICHED AI) ==
    # ==========================================
    with tab3:
        st.subheader("Causal Risk Simulator & Prescriptive Planner")
        
        # 1. WRAP ALL INPUTS IN A FORM TO PREVENT AUTO-RELOADS
        with st.form("simulation_form"):
            c1, c2 = st.columns(2)
            with c1:
                val_in = st.slider("💰 Order Value ($)", 0, 10000, 100) 
                weight_in = st.slider("⚖️ Weight (kg)", 0.0, 100.0, 2.0) 
                intl_in = st.checkbox("International Shipment", value=False)
            with c2:
                scenario_type = st.selectbox("Select Operational Scenario", ["Standard Operations (Default)", "Peak Season Stress (High Risk)", "Optimized Workflow (Low Risk)"])
                
                staff_str, vendor_val = ("Medium", 90) if "Standard" in scenario_type else (("Low", 60) if "Peak" in scenario_type else ("High", 100))
                
                c_sub3, c_sub4 = st.columns(2)
                with c_sub3: prod_in = st.selectbox("Product Category", encoders['product_type'].classes_, index=0)
                with c_sub4: mode_in = st.selectbox("Logistics Mode", encoders['shipping_mode'].classes_, index=1)

            # This button pauses Streamlit until clicked
            run_btn = st.form_submit_button("🚀 Run Prescriptive AI Analysis")

        # 2. ONLY EXECUTE THE ML AND AI IF THE BUTTON IS CLICKED
        if run_btn:
            sim_input = pd.DataFrame({
                'order_value': [val_in], 
                'package_weight_kg': [weight_in], 
                'is_international': [1 if intl_in else 0], 
                'is_large_electronic': [1 if prod_in == 'Large Electronic' else 0], 
                'staff_training_level': [encoders['staff_training_level'].transform([staff_str])[0]],
                'shipping_mode': [encoders['shipping_mode'].transform([mode_in])[0]], 
                'product_type': [encoders['product_type'].transform([prod_in])[0]],
                'vendor_reliability_score': [vendor_val] 
            })
            
            sim_prob = model.predict_proba(sim_input)[0][1]
            base_days = 2.0 if 'Air' in mode_in else (15.0 if 'Sea' in mode_in else 5.0)
            est_days = base_days + (5.0 if sim_prob > 0.5 else 0.0)
            
            st.markdown("---")
            res_c1, res_c2, res_c3 = st.columns([1.5, 1.5, 2])
            
            with res_c1:
                st.markdown("#### 🚦 Prediction")
                
                # --- UI FIX: Use the new helper function ---
                render_risk_badge(sim_prob)
                    
            with res_c2:
                st.markdown("#### ⏱️ Logistics Estimates")
                st.markdown(f"<h2 style='color:#f8fafc; margin:0; text-shadow: 0 0 20px rgba(255, 255, 255, 0.2);'>{est_days:.1f} Days</h2>", unsafe_allow_html=True)

            top_driver_list_tab3 = [] 
            with res_c3:
                st.markdown("#### 🧠 Primary Drivers")
                sim_vals = get_flat_shap(explainer, sim_input, len(X.columns))
                sim_impact = pd.DataFrame({'Factor': X.columns, 'Impact': sim_vals}).sort_values(by='Impact', ascending=False)
                
                risk_increasers = sim_impact[sim_impact['Impact'] > 0]
                if not risk_increasers.empty:
                    for i, row in risk_increasers.head(3).iterrows():
                        feat_col = row['Factor']
                        input_val = sim_input[feat_col].iloc[0]
                        display_name = feat_col.title().replace('_', ' ')
                        if feat_col == 'is_international': display_name = "International Shipment" if input_val == 1 else "Domestic Shipment"
                        elif feat_col == 'is_large_electronic': display_name = "Electronics" if input_val == 1 else "Non-Electronics"

                        top_driver_list_tab3.append({"feature": display_name, "val": row['Impact']})

                        contrib = row['Impact']
                        num_icons = 5 if contrib > 0.25 else (4 if contrib > 0.20 else (3 if contrib > 0.15 else (2 if contrib > 0.10 else 1)))
                        st.write(f"{'🔥' * num_icons} **{display_name}** (+{contrib * 100:.1f}%)")
                else:
                    st.markdown("<div style='color: #94a3b8; font-style: italic; font-size: 0.9rem;'>No risk-increasing drivers detected.</div>", unsafe_allow_html=True)

            sim_metadata_dict = {
                'order_value': val_in,
                'package_weight_kg': weight_in,
                'is_international': 1 if intl_in else 0,
                'product_type': prod_in,
                'shipping_mode': mode_in,
                'staff_training_level': staff_str 
            }

            # --- ENRICHED AGENT LAYER ---
            st.markdown("---")
            col_ai_t3, col_act_t3 = st.columns([1.5, 1])
            
            with col_ai_t3:
                with st.spinner("Agents are debating the causal risks..."):
                    agent_html = generate_risk_narrative(sim_prob, sim_metadata_dict, top_driver_list_tab3)
                    st.markdown(agent_html, unsafe_allow_html=True)
                    
            with col_act_t3:
                # --- ACADEMIC FIX: Pass the SHAP factors into the autonomous agent ---
                run_autonomous_agent(sim_prob, sim_input, "tab3", top_factors=top_driver_list_tab3)

    # ==========================================
    # === TAB 4: AGENTIC BUSINESS REPORT =======
    # ==========================================
    with tab4:
        st.subheader("🤖 Executive Agentic Business Report")
        st.markdown("Filter by process sequence group, then select a specific order to generate a comprehensive, AI-driven profit & risk analysis.")
        
        # 1. Cohort Selection (Metrics removed for a cleaner UI)
        selected_path_t4 = st.selectbox("Select Process Sequence Group:", sorted(df['Process_Path_Group'].value_counts().index), key="tab4_group")
        group_df_t4 = df[df['Process_Path_Group'] == selected_path_t4].drop_duplicates(subset=['case_id'])
            
        st.markdown("---")
        st.markdown(f"#### 📋 Case List: {selected_path_t4}")
        
        # 2. Render the Filtered Case List
        report_df = group_df_t4[['case_id', 'product_type', 'order_value', 'shipping_mode', 'has_manual_review']].copy()
        report_df['Status'] = report_df['has_manual_review'].apply(lambda x: "⛔ Review" if (x==1 or str(x).upper()=='TRUE') else "✅ Low Risk")
        
        report_selection = st.dataframe(
            report_df.head(100), 
            use_container_width=True, hide_index=True, height=250, 
            on_select="rerun", selection_mode="single-row"
        )

        # 3. Generate the Report for the Selected Case
        if len(report_selection.selection.rows) > 0:
            selected_report_id = report_df.iloc[report_selection.selection.rows[0]]['case_id']
            st.markdown("---")
            st.markdown(f"### 📑 Analysis for {selected_report_id}")
            
            row_idx = df[df['case_id'] == selected_report_id].index[0]
            case_data = df.iloc[row_idx]
            X_case = X.iloc[[row_idx]]
            prob = model.predict_proba(X_case)[0][1]
            
            vals = get_flat_shap(explainer, X_case, len(X.columns))
            impact_df = pd.DataFrame({'Feature': X.columns, 'Contribution': vals}).sort_values(by='Contribution', ascending=False)
            
            # --- ACADEMIC FIX: Extract both the feature name AND the mathematical value for the agent ---
            top_drivers = [{"feature": row['Feature'], "val": row['Contribution']} for i, row in impact_df.head(3).iterrows() if row['Contribution'] > 0]
            
            metadata_dict = {
                'order_value': case_data['order_value'], 
                'package_weight_kg': case_data['package_weight_kg'],
                'is_international': case_data['is_international'], 
                'product_type': case_data['product_type'],
                'shipping_mode': case_data['shipping_mode'],
                'staff_training_level': "Low" if case_data['staff_training_level'] == 0 else ("Medium" if case_data['staff_training_level'] == 1 else "High")
            }

            with st.spinner("Agent is compiling the executive report..."):
                report_html = generate_detailed_business_report(selected_report_id, prob, metadata_dict, top_drivers)
                st.markdown(report_html, unsafe_allow_html=True)
                
            try:
                # --- ACADEMIC FIX: Pass the SHAP factors into the autonomous agent ---
                run_autonomous_agent(prob, X_case, "tab4", top_factors=top_drivers)
            except NameError:
                pass 
        else:
            st.info("👆 Select a case from the list above to generate a 1-minute executive report.")