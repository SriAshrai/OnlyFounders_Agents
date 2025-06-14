import asyncio
import json
from typing import Dict, Any, List, Literal, Optional

# Correct Pydantic import
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

# Import the individual agent applications using correct (relative) paths
# Assuming:
# - Your Pitch Strength agent file is named 'Pitch_Strength_Langgraph.py'
# - Your Social Trust Graph agent file is named 'social_trust_graph_agent.py'
from agent_pitch_strength.Pitch_Strength_Langgraph import run_pitch_strength_agent as run_ps_agent_workflow
# --- CRITICAL CORRECTION: Corrected import path for Social Trust Graph Agent ---
from agent_social_trust.trust_graph_agent import run_social_trust_graph_agent as run_stg_agent_workflow

# --- 1. Define Combined Agent State ---
class CombinedAgentState(BaseModel):
    """
    Represents the combined state for the overall multi-agent workflow.
    """
    user_session_id: str = Field(description="Unique identifier for the user's session.")
    pitch_input_content: Optional[str] = Field(None, description="Raw text content of the pitch from user.")
    pitch_input_file_path: Optional[str] = Field(None, description="Path to the pitch file from user.")

    # Outputs from Pitch Strength Agent
    ps_agent_result: Optional[Dict[str, Any]] = Field(None, description="Results from Pitch Strength Agent.")

    # Inputs for Social Trust Graph Agent (derived from PS Agent, or user input)
    stg_input_data: List[Dict[str, Any]] = Field(default_factory=list, description="Data to feed into Social Trust Graph Agent.")
    # Outputs from Social Trust Graph Agent
    stg_agent_result: Optional[Dict[str, Any]] = Field(None, description="Results from Social Trust Graph Agent.")

    overall_status: Literal["pending", "pitch_analyzed", "graph_updated", "completed", "failed"] = "pending"
    overall_error: Optional[str] = Field(None, description="Overall error message for the combined workflow.")


# --- 2. Define Orchestration Nodes (Functions) ---

async def call_pitch_strength_agent_node(state: CombinedAgentState) -> CombinedAgentState:
    """
    Calls the Pitch Strength Agent workflow and captures its results.
    """
    print(f"\n--- Orchestrator: Calling Pitch Strength Agent for session {state.user_session_id} ---")
    try:
        # Generate a pitch_id unique to this pitch within the session
        pitch_id_for_ps = f"pitch_{state.user_session_id}"
        
        # Run the Pitch Strength Agent workflow
        ps_results = await run_ps_agent_workflow(
            pitch_id=pitch_id_for_ps,
            pitch_content=state.pitch_input_content,
            file_path=state.pitch_input_file_path
        )
        print(f"--- Orchestrator: Pitch Strength Agent returned: {json.dumps(ps_results, indent=2)}")

        if ps_results and ps_results.get("status") == "failed": # Check if ps_results is not None before .get()
            return {
                "overall_status": "failed",
                "overall_error": f"Pitch Strength Agent failed: {ps_results.get('error', 'Unknown error')}",
                "ps_agent_result": ps_results
            }
        
        print(f"--- Orchestrator: Pitch Strength Agent completed for session {state.user_session_id} ---")
        return {
            "overall_status": "pitch_analyzed",
            "ps_agent_result": ps_results
        }
    except Exception as e:
        print(f"Error calling Pitch Strength Agent: {e}")
        return {
            "overall_status": "failed",
            "overall_error": f"Orchestrator error calling PS Agent: {e}"
        }

async def prepare_stg_input_node(state: CombinedAgentState) -> CombinedAgentState:
    """
    Prepares input data for the Social Trust Graph Agent based on PS Agent's output.
    This is where intelligent connections between agents are made.
    """
    print(f"\n--- Orchestrator: Preparing STG input for session {state.user_session_id} ---")
    if state.overall_status == "failed" or not state.ps_agent_result:
        print("[Orchestrator] Skipping STG input preparation due to prior failure or missing PS Agent results.")
        return {"overall_status": state.overall_status, "overall_error": state.overall_error or "Missing PS Agent results."}

    # CONCEPTUAL LINKAGE:
    # Here, we infer a "project creation" or "founder participation" event from the pitch analysis.
    
    project_entity_id = f"project_{state.ps_agent_result['pitch_id']}" if state.ps_agent_result and 'pitch_id' in state.ps_agent_result else "unknown_project"
    founder_entity_id = f"founder_{state.user_session_id.split('_')[-1]}" # Derives a simple founder ID

    stg_data = []
    # Only generate STG data if PS Agent provided a valid overall_score
    if state.ps_agent_result and state.ps_agent_result.get("overall_score") is not None and state.ps_agent_result["overall_score"] > 0: # Ensure score is not None and greater than 0
        pitch_strength_score_scaled = state.ps_agent_result["overall_score"] / 100.0 # Scale to 0-1 for strength

        stg_data.append({
            "type": "endorsement",
            "endorser_id": founder_entity_id, 
            "endorsed_id": project_entity_id,
            "strength": pitch_strength_score_scaled,
            "reason": "PitchStrengthAgent_OverallScore"
        })
        stg_data.append({
            "type": "collaboration",
            "entity1_id": founder_entity_id,
            "entity2_id": project_entity_id,
            "weight": 1.0,
            "reason": "ProjectInitiation"
        })
    else:
        print(f"[Orchestrator] PS Agent overall_score was {state.ps_agent_result.get('overall_score')}. No STG data generated.")

    print(f"[Orchestrator] Prepared STG input data: {json.dumps(stg_data, indent=2)}")
    return {
        "stg_input_data": stg_data
    }

async def call_social_trust_graph_agent_node(state: CombinedAgentState) -> CombinedAgentState:
    """
    Calls the Social Trust Graph Agent workflow and captures its results.
    """
    print(f"\n--- Orchestrator: Calling Social Trust Graph Agent for session {state.user_session_id} ---")
    if state.overall_status == "failed":
        print("[Orchestrator] Skipping Social Trust Graph Agent due to prior failure.")
        return {
            "overall_status": state.overall_status, 
            "overall_error": state.overall_error,
            "stg_agent_result": {"status": "skipped", "error": "Previous agent failed."}
        }
    if not state.stg_input_data:
        print("[Orchestrator] Skipping Social Trust Graph Agent: No input data provided.")
        return {
            "overall_status": state.overall_status, # Preserve existing status
            "overall_error": state.overall_error,
            "stg_agent_result": {"status": "skipped", "error": "No input data provided."}
        }

    try:
        stg_session_id = f"stg_session_{state.user_session_id}"
        
        stg_results = await run_stg_agent_workflow(
            session_id=stg_session_id,
            new_graph_data=state.stg_input_data
        )
        print(f"--- Orchestrator: Social Trust Graph Agent returned: {json.dumps(stg_results, indent=2)}")

        if stg_results and stg_results.get("status") == "failed": 
            return {
                "overall_status": "failed",
                "overall_error": f"Social Trust Graph Agent failed: {stg_results.get('error', 'Unknown error')}",
                "stg_agent_result": stg_results
            }

        print(f"--- Orchestrator: Social Trust Graph Agent completed for session {state.user_session_id} ---")
        return {
            "overall_status": "graph_updated",
            "stg_agent_result": stg_results
        }
    except Exception as e:
        print(f"Error calling Social Trust Graph Agent: {e}")
        return {
            "overall_status": "failed",
            "overall_error": f"Orchestrator error calling STG Agent: {e}"
        }

async def final_orchestrator_output_node(state: CombinedAgentState) -> Dict[str, Any]:
    """
    Aggregates and formats the final output from both agents,
    ensuring a consistent structure for the frontend.
    """
    print(f"\n--- Orchestrator: Finalizing combined output for session {state.user_session_id} ---")
    
    # Initialize default values to ensure structure is always present
    status = "completed"
    error = None
    pitch_analysis_results = {}
    social_trust_analysis_results = {}

    # Check for overall failure from the state first
    if state.overall_status == "failed":
        status = "failed"
        error = state.overall_error or "Overall orchestration failed due to an earlier step."

    # Populate pitch analysis results and check for sub-agent failure
    if state.ps_agent_result:
        pitch_analysis_results = state.ps_agent_result
        if pitch_analysis_results.get("status") == "failed":
            status = "failed"
            error = error or pitch_analysis_results.get("error", "Pitch Strength Agent failed.")
    else:
        # If PS agent didn't run or returned nothing, set default structure
        pitch_analysis_results = {
            "pitch_id": state.user_session_id, 
            "overall_score": None, 
            "component_scores": {},
            "analysis_results": {"components": {}},
            "privacy_flags": {"tee_processed": False, "zkp_hash": None},
            "status": "not_run", # Explicitly set if PS agent never ran
            "on_chain_tx_hash": None
        }
    
    # Populate social trust analysis results and check for sub-agent failure/skipped status
    if state.stg_agent_result:
        social_trust_analysis_results = state.stg_agent_result
        if social_trust_analysis_results.get("status") == "failed":
            status = "failed"
            error = error or social_trust_analysis_results.get("error", "Social Trust Graph Agent failed.")
        # If STG was skipped, it's not a failure, the overall status remains what it was before STG
        # unless an explicit failure from PS agent or orchestration
    else:
        # If STG agent didn't run or returned nothing, set default structure
        social_trust_analysis_results = {
            "session_id": state.user_session_id,
            "overall_trust_scores": {},
            "current_graph_hash": None,
            "privacy_flags": {"zk_verified_endorsements_processed": False, "tee_analysis_conducted": False},
            "status": "not_run", # Explicitly set if STG agent never ran
            "on_chain_tx_hash": None
        }

    # If no explicit failure, ensure final status is 'completed'
    if status != "failed":
        status = "completed"

    final_response = {
        "status": status,
        "session_id": state.user_session_id,
        "error": error,
        "pitch_analysis": pitch_analysis_results,
        "social_trust_analysis": social_trust_analysis_results
    }
    
    print(f"--- Orchestrator: Final output for session {state.user_session_id}: {json.dumps(final_response, indent=2)} ---")
    return final_response


# --- 3. Build the Orchestrator LangGraph Workflow ---

combined_workflow = StateGraph(CombinedAgentState)

# Add nodes
combined_workflow.add_node("call_ps_agent", call_pitch_strength_agent_node)
combined_workflow.add_node("prepare_stg_input", prepare_stg_input_node)
combined_workflow.add_node("call_stg_agent", call_social_trust_graph_agent_node)
combined_workflow.add_node("final_output", final_orchestrator_output_node)

# Define entry point
combined_workflow.set_entry_point("call_ps_agent")

# Define edges
combined_workflow.add_edge("call_ps_agent", "prepare_stg_input")
combined_workflow.add_edge("prepare_stg_input", "call_stg_agent")
combined_workflow.add_edge("call_stg_agent", "final_output")
combined_workflow.add_edge("final_output", END)

# Compile the graph
orchestrator_app = combined_workflow.compile()

# --- 4. Function to run the overall orchestration ---
async def run_overall_orchestration(user_session_id: str, pitch_content: Optional[str] = None, file_path: Optional[str] = None):
    """
    Initiates the overall multi-agent orchestration and returns the final result.
    """
    initial_state = CombinedAgentState(
        user_session_id=user_session_id,
        pitch_input_content=pitch_content,
        file_path=file_path
    )
    print(f"\n\n===== Starting Overall Orchestration for Session ID: {user_session_id} =====")
    
    try:
        final_output_from_node = await orchestrator_app.ainvoke(initial_state, {"recursion_limit": 100})
        print(f"===== Finished Overall Orchestration for Session ID: {user_session_id} =====")
        return final_output_from_node
    except Exception as e:
        print(f"Error during overall orchestration for session {user_session_id}: {e}")
        return {
            "status": "failed",
            "session_id": user_session_id,
            "error": f"Orchestration failed: {e}",
            "pitch_analysis": {}, 
            "social_trust_analysis": {}
        }


if __name__ == "__main__":
    import os
    # Ensure nlp_processor.py and graph_processor.py exist in their respective directories
    # for this to run locally. Also ensure .env with GEMINI_API_KEY is in parent.

    dummy_pitch_for_orchestration = """
    Our new project, "Decentralized Governance Protocol (DGP)," aims to empower DAOs with truly liquid democracy.
    The team includes Dr. Ava Li (PhD in distributed systems, ex-ConsenSys) and Mr. Ben Carter (10 years in community management for large open-source projects).
    We identified a severe problem with voter apathy and whale dominance in current DAO structures. Our solution uses
    on-chain delegation and reputation-weighted voting to increase participation by 40% and fairness by 20%.
    The market for DAO tooling and governance solutions is rapidly maturing, projected to reach $10 billion by 2028.
    We have early commitments from three major DeFi protocols for pilot programs.
    """
    dummy_file_path_orch = "orchestrated_pitch_demo.txt"
    with open(dummy_file_path_orch, "w") as f:
        f.write(dummy_pitch_for_orchestration)

    asyncio.run(run_overall_orchestration("user_session_001", pitch_content=dummy_pitch_for_orchestration))

    os.remove(dummy_file_path_orch)
