import streamlit as st
import os
from cerebras.cloud.sdk import Cerebras
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Cerebras client
@st.cache_resource
def get_cerebras_client():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        st.error("CEREBRAS_API_KEY not found in environment variables")
        st.stop()
    return Cerebras(api_key=api_key)

client = get_cerebras_client()

# ==========================================
# DEMO SCENARIOS
# ==========================================
DEMO_SCENARIOS = {
    "Software Agency Crisis": {
        "business_type": "High-end Software Agency",
        "current_rule": "All client change requests must be approved by both the Founder (Revenue Protector) and Lead Architect (Quality Gatekeeper).",
        "scenario": "A major client threatens to walk away if a high-risk/high-reward feature is not delivered in 7 days. The feature requires bypassing standard QA protocols."
    },
    "Logistics Company Fuel Crisis": {
        "business_type": "Mid-sized Regional Logistics/Trucking Company",
        "current_rule": "Strict driver safety policy: Mandatory 10-hour rest break for every 14 hours on duty. No exceptions.",
        "scenario": "Fuel prices spike 40%. Major client demands 'Guaranteed 24-Hour Delivery' with strict penalty clauses. Drivers are pressured to skip rest breaks."
    },
    "Healthcare Staffing Shortage": {
        "business_type": "Regional Hospital Network",
        "current_rule": "Mandatory nurse-to-patient ratios (1:4 in general care, 1:2 in ICU). No nurse can work more than 12-hour shifts.",
        "scenario": "Flu outbreak causes 30% staff shortage. Emergency room wait times hit 8 hours. Administration considers suspending ratio requirements."
    },
    "E-commerce Black Friday": {
        "business_type": "Fast-growing E-commerce Startup",
        "current_rule": "All marketing campaigns must be approved by Legal team for compliance (data privacy, accessibility, claims verification).",
        "scenario": "Black Friday is in 48 hours. Marketing team has explosive viral campaign ready but Legal hasn't approved it. Competitors are already running similar campaigns."
    },
    "Manufacturing Quality Crisis": {
        "business_type": "Electronics Manufacturing Plant",
        "current_rule": "Zero-defect policy: Any product with defects must be scrapped. Quality inspectors have authority to halt production lines.",
        "scenario": "Major retailer threatens to cancel $10M order if shipment is delayed by even 1 day. Current defect rate is 3% (normally 0.5%). Production manager wants to ship anyway."
    }
}

# ==========================================
# JSON SCHEMAS (Strict Adherence)
# ==========================================

SPECTRUM_SCHEMA = {
    "type": "object",
    "properties": {
        "critical_conflict": {
            "type": "object",
            "properties": {
                "psychological_prior": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "scale_description": {"type": "string"}
                    },
                    "required": ["name", "scale_description"],
                    "additionalProperties": False
                },
                "operational_constraint": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "description"],
                    "additionalProperties": False
                }
            },
            "required": ["psychological_prior", "operational_constraint"],
            "additionalProperties": False
        },
        "agent_spectrum": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "prior_score": {"type": "integer"},
                    "persona": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["agent_id", "prior_score", "persona", "description"],
                "additionalProperties": False
            }
        }
    },
    "required": ["critical_conflict", "agent_spectrum"],
    "additionalProperties": False
}

SIMULATION_SCHEMA = {
    "type": "object",
    "properties": {
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time_step": {"type": "string"},
                    "agent_activity": {"type": "string"},
                    "approval_status": {"type": "string"},
                    "system_risk_score": {"type": "integer"},
                    "narrative_outcome": {"type": "string"}
                },
                "required": ["time_step", "agent_activity", "approval_status", "system_risk_score", "narrative_outcome"],
                "additionalProperties": False
            }
        },
        "tipping_point": {
            "type": "object",
            "properties": {
                "day": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["day", "description"],
            "additionalProperties": False
        }
    },
    "required": ["timeline", "tipping_point"],
    "additionalProperties": False
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "structural_patterns": {
            "type": "object",
            "properties": {
                "fracture": {"type": "string"},
                "tipping_point": {"type": "string"},
                "mechanism_of_failure": {"type": "string"}
            },
            "required": ["fracture", "tipping_point", "mechanism_of_failure"],
            "additionalProperties": False
        },
        "diagnosis": {
            "type": "object",
            "properties": {
                "mechanism_failure": {"type": "string"},
                "driver_failure": {"type": "string"}
            },
            "required": ["mechanism_failure", "driver_failure"],
            "additionalProperties": False
        },
        "proposed_solutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "solution_name": {"type": "string"},
                    "description": {"type": "string"},
                    "implementation": {"type": "string"},
                    "prescriptive_action_plan": {"type": "string"}
                },
                "required": ["solution_name", "description", "implementation", "prescriptive_action_plan"],
                "additionalProperties": False
            }
        }
    },
    "required": ["structural_patterns", "diagnosis", "proposed_solutions"],
    "additionalProperties": False
}

def call_cerebras(messages, schema, schema_name):
    """Call Cerebras API with strict structured output"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema
                }
            }
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Error calling Cerebras API: {str(e)}")
        return None

# ==========================================
# STATE MANAGEMENT
# ==========================================
if 'phase' not in st.session_state:
    st.session_state.phase = 1
if 'business_context' not in st.session_state:
    st.session_state.business_context = {}
if 'spectrum_data' not in st.session_state:
    st.session_state.spectrum_data = None
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'selected_demo' not in st.session_state:
    st.session_state.selected_demo = None

# ==========================================
# UI LAYOUT
# ==========================================
st.set_page_config(page_title="Stress-Test Engine", layout="wide")
st.title("🎯 Business Stress-Test Engine")
st.markdown("*A structural analysis framework for identifying organizational tipping points.*")

# Dynamic Navigation Indicators
# Phase 1 is done if phase > 1
p1_icon = "✅" if st.session_state.phase > 1 else "1️⃣"
# Phase 2 is done if phase > 2
p2_icon = "✅" if st.session_state.phase > 2 else "2️⃣" if st.session_state.phase == 2 else "⏸️"
# Phase 3 is "Done" (Checked) if analysis exists, otherwise it's Active (3) or Pending
p3_icon = "✅" if st.session_state.analysis_data else "3️⃣" if st.session_state.phase == 3 else "⏸️"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**{p1_icon} Phase 1: Spectrum**")
    if st.session_state.phase > 1:
        st.caption("Context & Agents Defined")
with col2:
    st.markdown(f"**{p2_icon} Phase 2: Simulation**")
    if st.session_state.phase > 2:
        st.caption("Dynamics Simulated")
with col3:
    st.markdown(f"**{p3_icon} Phase 3: Analysis**")
    if st.session_state.analysis_data:
        st.caption("Insights Generated")

st.divider()

# ==========================================
# PHASE 1: CONTEXT & SPECTRUM
# ==========================================
if st.session_state.phase == 1:
    st.header("Phase 1: Context & Spectrum Injection")
    st.markdown("Define your business scenario to identify critical psychological priors and operational constraints.")
    
    # Demo Scenario Selection
    st.subheader("📚 Try a Demo Scenario")
    demo_cols = st.columns(5)
    
    for idx, (demo_name, demo_data) in enumerate(DEMO_SCENARIOS.items()):
        with demo_cols[idx]:
            if st.button(demo_name, key=f"demo_{idx}", use_container_width=True):
                st.session_state.selected_demo = demo_data
                st.rerun()
    
    st.divider()
    
    # Initialize form values
    default_business = ""
    default_rule = ""
    default_scenario = ""
    
    # If a demo was selected, use its values
    if st.session_state.selected_demo:
        default_business = st.session_state.selected_demo["business_type"]
        default_rule = st.session_state.selected_demo["current_rule"]
        default_scenario = st.session_state.selected_demo["scenario"]
        # Clear the selection after using it
        st.session_state.selected_demo = None
    
    col_a, col_b = st.columns(2)
    with col_a:
        business_type = st.text_input("Business Type", value=default_business, placeholder="e.g., High-end Software Agency, Regional Hospital Network")
        current_rule = st.text_area(
            "Current Operational Rule/Constraint",
            value=default_rule,
            placeholder="Describe the existing policy or process that will be tested...",
            height=100
        )
    with col_b:
        scenario = st.text_area(
            "Crisis Scenario",
            value=default_scenario,
            placeholder="Describe the specific crisis situation and pressures...",
            height=185
        )
    
    if st.button("🔬 Generate Spectrum Analysis", type="primary", disabled=not (business_type and current_rule and scenario)):
        with st.spinner("Analyzing scenario and generating agent spectrum..."):
            prompt = f"""You are a Complexity Analyst. Analyze the following business scenario.

Business Type: {business_type}
Current Rule: {current_rule}
Crisis: {scenario}

Task:
1. Identify the most critical psychological Prior (belief) and Operational Constraint in conflict.
2. Generate a spectrum of 10 agents with a bimodal distribution (4 high adherence, 4 low adherence, 2 neutral).
3. Assign each agent a persona and prior score (1-10 scale).

Output must strictly follow the JSON schema."""

            messages = [
                {"role": "system", "content": "You are a business systems analyst. Generate realistic agent profiles."},
                {"role": "user", "content": prompt}
            ]
            
            result = call_cerebras(messages, SPECTRUM_SCHEMA, "spectrum_schema")
            
            if result:
                st.session_state.spectrum_data = result
                st.session_state.business_context = {
                    "business_type": business_type,
                    "scenario": scenario,
                    "current_rule": current_rule
                }
                st.session_state.phase = 2
                st.rerun()

# ==========================================
# PHASE 2: DYNAMIC SIMULATION
# ==========================================
elif st.session_state.phase == 2:
    st.header("Phase 2: Dynamic Simulation")
    
    if st.session_state.spectrum_data:
        # Spectrum Visualization
        conflict = st.session_state.spectrum_data['critical_conflict']
        
        st.info(f"**Conflict Detected:** {conflict['psychological_prior']['name']} vs. {conflict['operational_constraint']['name']}")
        
        with st.expander("📊 View Agent Spectrum Details", expanded=True):
            # Create a dataframe for the chart
            agents = st.session_state.spectrum_data['agent_spectrum']
            df_agents = pd.DataFrame(agents)
            
            # Simple bar chart of Prior Scores
            st.bar_chart(df_agents.set_index('agent_id')['prior_score'])
            
            # List details
            for agent in agents:
                st.markdown(f"**{agent['agent_id']} ({agent['persona']})**: {agent['description']} [Score: {agent['prior_score']}]")
        
        st.divider()
        
        simulation_days = st.slider("Simulation Duration (Days)", 3, 7, 5)
        
        if st.button("▶️ Run Simulation", type="primary"):
            with st.spinner(f"Running {simulation_days}-day simulation..."):
                context = st.session_state.business_context
                spectrum = st.session_state.spectrum_data
                
                prompt = f"""Run a {simulation_days}-day simulation.

Context: {context['business_type']} | {context['scenario']}
Rule: {context['current_rule']}
Conflict: {spectrum['critical_conflict']['psychological_prior']['name']} vs {spectrum['critical_conflict']['operational_constraint']['name']}

Agents:
{json.dumps(spectrum['agent_spectrum'], indent=2)}

Task:
Simulate interactions over {simulation_days} days.
Track:
1. Activity & Decisions
2. Approval Status (Approved/Delayed/Bypassed)
3. System Risk Score (Start 0. Add +2 for protests/delays, +10 for bypasses/violations).
4. Narrative outcome.

Identify the specific Tipping Point day."""

                messages = [
                    {"role": "system", "content": "You are a dynamic simulation engine. Generate time-step narratives showing risk accumulation."},
                    {"role": "user", "content": prompt}
                ]
                
                result = call_cerebras(messages, SIMULATION_SCHEMA, "simulation_schema")
                
                if result:
                    st.session_state.simulation_data = result
                    st.session_state.phase = 3
                    st.rerun()

# ==========================================
# PHASE 3: ANALYSIS & VISUALIZATION
# ==========================================
elif st.session_state.phase == 3:
    st.header("Phase 3: Structural Insight & Solutions")
    
    # ----------------------------------------------------
    # SECTION A: SIMULATION VISUALIZATION (Context)
    # ----------------------------------------------------
    if st.session_state.simulation_data:
        sim = st.session_state.simulation_data
        
        st.subheader("📉 Step 1: Simulation Visualization")
        st.markdown("Review the simulation log to verify the tipping point before running structural analysis.")
        
        # Prepare data for Risk Chart
        timeline_data = []
        for event in sim['timeline']:
            timeline_data.append({
                "Day": event['time_step'],
                "Risk Score": event['system_risk_score']
            })
        df_risk = pd.DataFrame(timeline_data)
        
        col_chart, col_stats = st.columns([2, 1])
        
        with col_chart:
            st.caption("System Risk Score Trajectory")
            st.line_chart(df_risk.set_index("Day"))
            
        with col_stats:
            st.metric("Final Risk Score", df_risk["Risk Score"].iloc[-1])
            st.error(f"Tipping Point: {sim['tipping_point']['day']}")
            st.caption(sim['tipping_point']['description'])

        with st.expander("📖 View Full Simulation Log"):
            for event in sim['timeline']:
                st.markdown(f"**{event['time_step']}** | Status: `{event['approval_status']}`")
                st.write(event['narrative_outcome'])
                st.divider()
        
        st.divider()

    # ----------------------------------------------------
    # SECTION B: STRUCTURAL ANALYSIS (Results)
    # ----------------------------------------------------
    st.subheader("🧠 Step 2: Structural Analysis")

    # If analysis exists, show results. If not, show button.
    if not st.session_state.analysis_data:
        st.info("The simulation has identified a fracture point. Click below to apply the Structural Toolkit to diagnose root causes and generate fixes.")
        
        if st.button("🔍 Generate Structural Analysis", type="primary"):
            with st.spinner("Applying Structural Toolkit..."):
                prompt = f"""Analyze this simulation log.

Business: {st.session_state.business_context['business_type']}
Simulation: {json.dumps(st.session_state.simulation_data, indent=2)}

Task:
1. Extract structural patterns (fracture, tipping point, failure mechanism).
2. Diagnose mechanism and driver failures.
3. Propose structural solutions. 
CRITICAL: For each solution, provide a 'prescriptive_action_plan' containing specific metrics, bonuses, or exact rule changes (The 'How-To' layer)."""

                messages = [
                    {"role": "system", "content": "You are a structural analyst. Provide deep system diagnostics and prescriptive, actionable fixes."},
                    {"role": "user", "content": prompt}
                ]
                
                result = call_cerebras(messages, ANALYSIS_SCHEMA, "analysis_schema")
                
                if result:
                    st.session_state.analysis_data = result
                    st.rerun()

    # DISPLAY ANALYSIS RESULTS
    if st.session_state.analysis_data:
        analysis = st.session_state.analysis_data
        
        # Pattern Extraction
        st.markdown("#### 🔬 Structural Pattern Extraction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pattern", "Fracture")
        c1.write(analysis['structural_patterns']['fracture'])
        
        c2.metric("Critical Moment", "Tipping Point")
        c2.write(analysis['structural_patterns']['tipping_point'])
        
        c3.metric("Root Cause", "Mechanism")
        c3.write(analysis['structural_patterns']['mechanism_of_failure'])
        
        st.divider()

        # Diagnosis
        st.markdown("#### 🩺 Diagnosis")
        d1, d2 = st.columns(2)
        with d1:
            st.error("**Mechanism Failure**")
            st.write(analysis['diagnosis']['mechanism_failure'])
        with d2:
            st.warning("**Driver Failure**")
            st.write(analysis['diagnosis']['driver_failure'])
            
        st.divider()
        
        # Solutions
        st.markdown("#### 💡 Proposed Structural Mutations")
        
        for i, solution in enumerate(analysis['proposed_solutions'], 1):
            with st.container():
                st.markdown(f"**{i}. {solution['solution_name']}**")
                st.write(f"*Strategy:* {solution['description']}")
                st.write(f"*Implementation:* {solution['implementation']}")
                st.info(f"**🛠️ Prescriptive Action Plan:** {solution['prescriptive_action_plan']}")
                st.divider()
        
        # Footer Actions
        col_export, col_reset = st.columns(2)
        with col_export:
            full_report = {
                "context": st.session_state.business_context,
                "spectrum": st.session_state.spectrum_data,
                "simulation": st.session_state.simulation_data,
                "analysis": st.session_state.analysis_data
            }
            st.download_button(
                "📥 Download Full Stress-Test Report",
                data=json.dumps(full_report, indent=2),
                file_name="stress_test_report.json",
                mime="application/json"
            )
        with col_reset:
            if st.button("🔄 Start New Scenario"):
                for key in ['spectrum_data', 'simulation_data', 'analysis_data']:
                    st.session_state[key] = None
                st.session_state.phase = 1
                st.rerun()

st.divider()
st.markdown(
    """
    <div style="text-align:center; font-weight:700; font-size:16px; padding:10px; background:#f6f8fa; border-radius:6px;">
      Built by <a href="http://kongokega.com/" target="_blank">kongokega.com</a> —
      <a href="https://www.linkedin.com/in/vincent-mbogo/" target="_blank">LinkedIn: Vincent Mbogo</a> —
      <a href="https://x.com/kongokega" target="_blank">X: @kongokega</a>
    </div>
    """,
    unsafe_allow_html=True
)