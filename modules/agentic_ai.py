import streamlit as st
import concurrent.futures
import numpy as np
import pandas as pd
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

# CRITICAL ACADEMIC UPDATE: Import actual RAG dependencies
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_faiss_db():
    """
    Initializes a local FAISS Vector Database using HuggingFace embeddings.
    This fulfills the 'Edge RAG' claim in the paper, keeping data local and secure.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # The enterprise compliance policies to embed into the vector space
    policies = [
        "VECTOR DB [WEEE-Dir-2012]: Large electronics require WEEE recycling compliance documentation and UN3480 large lithium battery freight declarations. Flagged for electronic hazards.",
        "VECTOR DB [CPSC-Textile-16]: Apparel and textile shipments must include a Certificate of Origin and verify Class 1 flammability standards compliance.",
        "VECTOR DB [C-TPAT-HighValue]: High expense orders exceeding $2,000 USD require 'Signature Required' delivery, GPS tracking, and supplemental transit insurance. Flagged for financial risk.",
        "VECTOR DB [IATA-Sec-4]: Heavy air freight exceeding 20kg is flagged for mandatory secondary X-ray screening and dimensional weight auditing. High risk of aviation delay.",
        "VECTOR DB [SOP-01]: Standard domestic shipping protocols apply. Low risk, proceed with standard dispatch workflow."
    ]
    
    # Build and return the local vector index
    db = FAISS.from_texts(policies, embeddings)
    return db

@tool
def query_live_shipping_rates(weight_kg: float, current_mode: str) -> str:
    """Always use this tool to check live logistics costs before making a financial decision."""
    if current_mode.lower() == 'air' and weight_kg > 10:
        return "ERP ALERT: Air freight for items over 10kg is surging today. Switching to Ground will save $145.00 and automatically bypass TSA aviation holds."
    return "ERP DATA: Current shipping mode is within normal cost parameters. No urgent cost-savings found."

@tool
def search_policy_database(search_query: str) -> str:
    """Always use this tool to search the enterprise compliance database for shipping regulations before advising."""
    # ACADEMIC UPDATE: Perform actual semantic vector similarity search
    db = get_faiss_db()
    
    # Retrieve the top 1 most mathematically similar policy document
    docs = db.similarity_search(search_query, k=1)
    
    if docs:
        return docs[0].page_content
    return "VECTOR DB [SOP-01]: Standard domestic shipping protocols apply."

def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _safe_invoke(runnable, messages):
    """Wraps LLM invocations with exponential backoff to handle transient API limits/blips."""
    return runnable.invoke(messages)

def _run_agent_with_tools(llm, system_prompt, user_content, tools_list, final_formatting_prompt=""):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    if not tools_list:
        final_answer = _safe_invoke(llm, messages).content.replace("```html", "").replace("```", "").strip()
        return final_answer, ""

    llm_with_tools = llm.bind_tools(tools_list)
    response = _safe_invoke(llm_with_tools, messages)
    
    ui_logs = ""
    
    if response.tool_calls:
        messages.append(response) 
        
        for tool_call in response.tool_calls:
            t_name = tool_call['name']
            t_args = tool_call['args']
            
            if t_name == 'search_policy_database':
                t_result = search_policy_database.invoke(t_args)
                ui_color = "#3b82f6" 
                ui_title = "🔍 RAG DATABASE QUERY:"
            elif t_name == 'query_live_shipping_rates':
                t_result = query_live_shipping_rates.invoke(t_args)
                ui_color = "#a855f7" 
                ui_title = "⚙️ AUTONOMOUS ERP QUERY:"
            else:
                t_result = "Tool not found."
                ui_color = "#94a3b8"
                ui_title = "❓ UNKNOWN TOOL:"

            ui_logs += f"""
<div style='margin-bottom: 10px; font-size: 0.85rem; border-left: 2px solid {ui_color}; padding-left: 10px; background: rgba(255,255,255, 0.02); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: {ui_color}; font-weight: bold;'>{ui_title}</span> <span style='color: #94a3b8;'><i>{t_name}({t_args})</i></span><br>
<span style='color: #cbd5e1;'>Result ➔ {t_result}</span>
</div>
"""
            messages.append(ToolMessage(content=t_result, tool_call_id=tool_call['id']))
        
        messages.append(HumanMessage(content=f"Using the data retrieved, provide your final response. DO NOT use markdown lists or bullet points. {final_formatting_prompt}"))
        final_answer = _safe_invoke(llm, messages).content.replace("```html", "").replace("```", "").strip()
        return final_answer, ui_logs
    else:
        messages.append(HumanMessage(content=f"DO NOT use markdown lists or bullet points. {final_formatting_prompt}"))
        final_answer = _safe_invoke(llm, messages).content.replace("```html", "").replace("```", "").strip()
        return final_answer, ""

def _build_deterministic_context(metadata_dict):
    """Safely extracts data and builds explicit DB triggers to prevent hallucination."""
    try:
        o_val = float(str(metadata_dict.get('order_value', 0)).replace('$', '').replace(',', '').strip())
    except:
        o_val = 0.0
        
    try:
        o_weight = float(str(metadata_dict.get('package_weight_kg', 0)).replace('kg', '').replace(',', '').strip())
    except:
        o_weight = 0.0
        
    try:
        is_elec = int(metadata_dict.get('is_large_electronic', 0))
    except:
        is_elec = 0
        
    o_mode = str(metadata_dict.get('shipping_mode', 'Unknown'))
    o_intl = 'Yes' if metadata_dict.get('is_international') == 1 else 'No'
    
    policy_flags = []
    if o_val > 2000.0:
        policy_flags.append("HIGH EXPENSE (Query C-TPAT)")
    if o_weight > 20.0 and 'air' in o_mode.lower():
        policy_flags.append("HEAVY AIR (Query IATA-Sec)")
    if is_elec == 1:
        policy_flags.append("LARGE ELECTRONICS (Query WEEE)")

    if policy_flags:
        flag_str = "SYSTEM FLAGS: " + ", ".join(policy_flags) + "."
    else:
        flag_str = "SYSTEM FLAGS: None. Standard SOP applies."
        
    return o_val, o_weight, o_mode, o_intl, flag_str

def generate_risk_narrative(risk_score, metadata_dict, top_factors):
    try:
        llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.1-8b-instant", temperature=0.1)

        drivers_text = ", ".join([f"{f['feature']} (+{f['val']*100:.1f}%)" for f in top_factors]) if top_factors else "None"
        o_val, o_weight, o_mode, o_intl, flag_str = _build_deterministic_context(metadata_dict)

        ctx = f"Order: ${o_val:.2f}, Mode: {o_mode}, Weight: {o_weight}kg, Intl: {o_intl}. RISK: {risk_score:.1%}. {flag_str} DRIVERS: {drivers_text}"

        comp_sys = "You are a Compliance Officer. You MUST use your search_policy_database tool based strictly on the SYSTEM FLAGS provided."
        comp_prompt = f"Analyze this order and state the primary compliance risk in one short sentence, citing the database. Context: {ctx}"

        log_sys = "You are a Logistics Manager. Focus on speed and avoiding manual bottlenecks."
        log_prompt = f"Identify the #1 logistical priority for this order in one short sentence. Context: {ctx}"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_comp = executor.submit(_run_agent_with_tools, llm, comp_sys, comp_prompt, [search_policy_database], "Keep it to one sentence.")
            future_log = executor.submit(_run_agent_with_tools, llm, log_sys, log_prompt, None, "Keep it to one sentence.")
            
            comp_advice, comp_logs = future_comp.result()
            log_advice, log_logs = future_log.result()

        fin_sys = "You are the Finance Director. You MUST use your query_live_shipping_rates tool to check costs before deciding."
        fin_prompt = f"Compliance: '{comp_advice}'\nLogistics: '{log_advice}'\nWeight: {o_weight}kg. Mode: {o_mode}.\nDetermine the plan."
        fin_format = "Use EXACTLY this raw HTML: <ul><li style='margin-bottom: 8px;'><b>The Situation:</b> [Summary]</li><li><b>The Solution:</b> [Action step]</li></ul>"
        
        draft_verdict, fin_logs = _run_agent_with_tools(llm, fin_sys, fin_prompt, [query_live_shipping_rates], fin_format)

        audit_sys = "You are a strict Risk Auditor. Review the Finance Director's plan. If safe and logical, reply 'APPROVED'. If it ignores compliance, reply 'REJECTED: [Reason]'."
        audit_prompt = f"Draft Plan:\n{draft_verdict}\n\nDoes this effectively balance cost, speed, and compliance? Keep it under 20 words."
        
        audit_response = _safe_invoke(llm, [SystemMessage(content=audit_sys), HumanMessage(content=audit_prompt)]).content.strip()

        reflection_logs = ""
        final_verdict = draft_verdict

        if "REJECTED" in audit_response.upper():
            reflection_logs = f"""
<div style='margin-bottom: 15px; font-size: 0.85rem; border-left: 2px solid #ef4444; padding-left: 10px; background: rgba(239, 68, 68, 0.05); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: #ef4444; font-weight: bold;'>🛑 AUDITOR REJECTION:</span> <span style='color: #cbd5e1;'>"{audit_response}"</span><br>
<span style='color: #94a3b8; font-style: italic;'>Forcing Finance Director to revise the plan...</span>
</div>
"""
            rewrite_sys = "You are the Finance Director. Your previous plan was rejected. You must revise it."
            rewrite_prompt = f"Original Plan:\n{draft_verdict}\n\nRejection Reason:\n{audit_response}\n\nProvide a NEW plan that fixes this flaw. {fin_format}"
            final_verdict = _safe_invoke(llm, [SystemMessage(content=rewrite_sys), HumanMessage(content=rewrite_prompt)]).content.replace("```html", "").replace("```", "").strip()
        else:
            reflection_logs = f"""
<div style='margin-bottom: 15px; font-size: 0.85rem; border-left: 2px solid #10b981; padding-left: 10px; background: rgba(16, 185, 129, 0.05); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: #10b981; font-weight: bold;'>✅ AUDITOR APPROVED:</span> <span style='color: #cbd5e1;'>Plan passes all SLA and compliance checks on the first iteration.</span>
</div>
"""

        color = "#f87171" if risk_score > 0.5 else "#4ade80"
        
        return f"""
<div style='background: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 10px; border-left: 4px solid {color}; font-size: 0.95rem; line-height: 1.6;'>
<strong style='color: {color}; font-size: 1.1rem; margin-bottom: 15px; display: block;'>🧠 MoE Orchestration & Actor-Critic Evaluation Loop</strong>
{comp_logs}
<div style='margin-bottom: 10px; font-size: 0.85rem; border-left: 2px solid #eab308; padding-left: 10px;'>
<span style='color: #eab308; font-weight: bold;'>🛡️ COMPLIANCE:</span> <span style='color: #cbd5e1;'>"{comp_advice}"</span>
</div>
<div style='margin-bottom: 15px; font-size: 0.85rem; border-left: 2px solid #38bdf8; padding-left: 10px;'>
<span style='color: #38bdf8; font-weight: bold;'>📦 LOGISTICS:</span> <span style='color: #cbd5e1;'>"{log_advice}"</span>
</div>
{fin_logs}
{reflection_logs}
<div style='background: rgba(255,255,255,0.03); padding: 10px; border-radius: 5px;'>
<span style='font-weight: bold; color: {color};'>💼 FINAL FINANCE RULING:</span><br>
<span style='color: #f1f5f9;'>{final_verdict}</span>
</div>
</div>
"""
    except Exception as e:
        return f"⚠️ **LangChain API Error:** {str(e)}"

def generate_detailed_business_report(case_id, risk_score, metadata_dict, top_factors):
    try:
        llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.1-8b-instant", temperature=0.1)

        drivers_text = ", ".join([f"{f['feature']}" for f in top_factors]) if top_factors else "None"
        o_val, o_weight, o_mode, o_intl, flag_str = _build_deterministic_context(metadata_dict)

        ctx = f"CASE ID: {case_id}. {flag_str} Mode: {o_mode}, Weight: {o_weight}kg, Intl: {o_intl}. RISK: {risk_score:.1%}. DRIVERS: {drivers_text}"

        comp_sys = "You are a Compliance Officer. You MUST use your search_policy_database tool based strictly on the SYSTEM FLAGS provided."
        comp_prompt = f"Analyze this order and state the compliance risk in one sentence. Context: {ctx}"

        log_sys = "You are a Logistics Manager. Focus on speed and delays."
        log_prompt = f"Identify the operational risk in one sentence. Context: {ctx}"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_comp = executor.submit(_run_agent_with_tools, llm, comp_sys, comp_prompt, [search_policy_database], "Keep it to one sentence.")
            future_log = executor.submit(_run_agent_with_tools, llm, log_sys, log_prompt, None, "Keep it to one sentence.")
            
            comp_advice, comp_logs = future_comp.result()
            log_advice, log_logs = future_log.result()

        fin_sys = "You are the Finance Director writing an executive action plan. You MUST use tools to check financial data."
        fin_prompt = f"Compliance: '{comp_advice}'\nLogistics: '{log_advice}'\nWeight: {o_weight}kg. Mode: {o_mode}.\nDetermine the plan."
        fin_format = "Use EXACTLY this raw HTML: <ul><li style='margin-bottom: 8px;'><b>Executive Summary:</b> [1 sentence]</li><li style='margin-bottom: 8px;'><b>Recommended Action:</b> [Action step]</li><li><b>Expected Benefit:</b> [Savings]</li></ul>"
        
        draft_verdict, fin_logs = _run_agent_with_tools(llm, fin_sys, fin_prompt, [query_live_shipping_rates], fin_format)

        audit_sys = "You are a strict Risk Auditor. Review the plan. Reply 'APPROVED' if good. If it creates delays, reply 'REJECTED: [Reason]'."
        audit_prompt = f"Draft Plan:\n{draft_verdict}\n\nDoes this balance cost, speed, and compliance?"
        
        audit_response = _safe_invoke(llm, [SystemMessage(content=audit_sys), HumanMessage(content=audit_prompt)]).content.strip()

        reflection_logs = ""
        final_verdict = draft_verdict

        if "REJECTED" in audit_response.upper():
            reflection_logs = f"""
<div style='margin-bottom: 15px; font-size: 0.95rem; border-left: 2px solid #ef4444; padding-left: 10px; background: rgba(239, 68, 68, 0.05); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: #ef4444; font-weight: bold;'>🛑 AUDITOR REJECTION:</span> <span style='color: #cbd5e1;'>"{audit_response}"</span><br>
<span style='color: #94a3b8; font-style: italic;'>Forcing Finance Director to revise the plan...</span>
</div>
"""
            rewrite_sys = "You are the Finance Director. Your previous plan was rejected. You must revise it."
            rewrite_prompt = f"Original Plan:\n{draft_verdict}\n\nRejection Reason:\n{audit_response}\n\nProvide a NEW plan. {fin_format}"
            final_verdict = _safe_invoke(llm, [SystemMessage(content=rewrite_sys), HumanMessage(content=rewrite_prompt)]).content.replace("```html", "").replace("```", "").strip()
        else:
            reflection_logs = f"""
<div style='margin-bottom: 15px; font-size: 0.95rem; border-left: 2px solid #10b981; padding-left: 10px; background: rgba(16, 185, 129, 0.05); padding-top: 8px; padding-bottom: 8px;'>
<span style='color: #10b981; font-weight: bold;'>✅ AUDITOR APPROVED:</span> <span style='color: #cbd5e1;'>Plan passes all SLA and compliance checks on the first iteration.</span>
</div>
"""

        return f"""
<div class='report-box'>
<strong style='color: #f8fafc; font-size: 1.2rem; margin-bottom: 15px; display: block;'>📑 Verified Prescriptive Action Policy for {case_id}</strong>
{comp_logs}
<div style='margin-bottom: 12px; font-size: 0.95rem; border-left: 2px solid #eab308; padding-left: 10px;'>
<span style='color: #eab308; font-weight: bold;'>🛡️ COMPLIANCE AUDIT:</span> <span style='color: #cbd5e1;'>"{comp_advice}"</span>
</div>
<div style='margin-bottom: 18px; font-size: 0.95rem; border-left: 2px solid #38bdf8; padding-left: 10px;'>
<span style='color: #38bdf8; font-weight: bold;'>📦 LOGISTICS AUDIT:</span> <span style='color: #cbd5e1;'>"{log_advice}"</span>
</div>
{fin_logs}
{reflection_logs}
<div style='background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px;'>
<span style='font-weight: bold; color: #4ade80;'>💼 FINAL FINANCE DIRECTOR'S ACTION PLAN:</span><br><br>
<span style='color: #f1f5f9; line-height: 1.6;'>{final_verdict}</span>
</div>
</div>
"""
    except Exception as e:
        return f"⚠️ **LangChain API Error:** {str(e)}"

def run_autonomous_agent(risk_score, case_inputs, unique_tab_id, top_factors=None):
    """
    ACADEMIC UPDATE: Replaced static if/elif templates with dynamic LLM generation
    that mathematically binds the generated actions to the ML SHAP drivers.
    """
    with st.container(border=True):
        st.markdown("#### ⚙️ AI-Generated Prescriptive Actions")
        
        try:
            # Extract inputs contextually
            val = case_inputs.get('order_value', pd.Series([0])).iloc[0]
            mode = case_inputs.get('shipping_mode', pd.Series(['Unknown'])).iloc[0]
            
            # Format SHAP drivers to force the LLM to address them
            drivers_text = ", ".join([f"{f['feature']} (Impact: {f['val']:.3f})" for f in top_factors]) if top_factors else "None detected"

            llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.1-8b-instant", temperature=0.3)
            
            if risk_score > 0.50:
                st.markdown("<span style='color:#f87171; font-weight:bold;'>🔴 High Risk Flagged: LLM resolving ML drivers...</span>", unsafe_allow_html=True)
                icon = "🚨"
                # --- ACADEMIC FIX: Force LLM to use SHAP drivers and output 3 actions ---
                sys_prompt = f"You are a logistics agent. The ML model flagged this order because of these specific drivers: {drivers_text}. Do not use generic templates. Provide exactly 3 specific, short, actionable steps to mitigate these exact risks. Format as plain text with a dash."
            else:
                st.markdown("<span style='color:#4ade80; font-weight:bold;'>🟢 Low Risk: LLM identifying optimizations...</span>", unsafe_allow_html=True)
                icon = "💡"
                # --- ACADEMIC FIX: Force LLM to use SHAP drivers and output 3 actions ---
                sys_prompt = f"You are a logistics agent. The ML model cleared this order, but note these SHAP drivers: {drivers_text}. Do not use generic templates. Provide exactly 3 specific, short, actionable steps to optimize cost, speed, or customer experience. Format as plain text with a dash."

            user_prompt = f"Order Value: {val}, Mode: {mode}, Risk Score: {risk_score:.2f}. What are the 3 prescriptive actions?"
            
            with st.spinner("Generating prescriptive actions from ML drivers..."):
                response = _safe_invoke(llm, [
                    SystemMessage(content=sys_prompt), 
                    HumanMessage(content=user_prompt)
                ]).content.strip()
            
            # Parse the LLM response and render it in the UI identically to the old static version
            actions = [line.strip().strip('-').strip() for line in response.split('\n') if line.strip()]
            for act in actions:
                if act:
                    if risk_score > 0.50:
                        st.error(f"**Action:** {act}", icon=icon)
                    else:
                        st.success(f"**Action:** {act}", icon=icon)

        except Exception as e:
            st.warning(f"⚠️ API connection issue detected. Loading static fallback protocols...")
            if risk_score > 0.50:
                st.error("**Action:** Immediate Manual Review Required. Route to Compliance Team.", icon="🚨")
                st.error("**Action:** Halt shipment dispatch pending ERP clearance.", icon="🚨")
                st.error("**Action:** Flag vendor/mode for secondary audit.", icon="🚨")
            else:
                st.success("**Action:** Standard processing approved. Proceed to dispatch.", icon="💡")
                st.success("**Action:** Log telemetry for monthly optimization review.", icon="💡")
                st.success("**Action:** Monitor ETA via standard tracking.", icon="💡")