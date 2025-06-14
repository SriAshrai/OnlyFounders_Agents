import asyncio
import os
import hashlib # For ZKP hash mock and pitch ID
import json # For formatting LLM output and ZKP inputs
from typing import Dict, Any, List, Literal, Optional

# Correct Pydantic import
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# Correct import for LangChain's Google Generative AI integration
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END

# Correct relative import for nlp_processor (now initial_setup.py)
from .initial_setup import NLPProcessor

# --- 0. Configuration ---
# API Key will be loaded from .env
gemini_llm_model = "gemini-1.5-flash" # Updated to a latest model for better availability
embedding_model_name = "models/text-embedding-004" # For originality analysis


# --- 1. Define Agent State ---
class PitchStrengthAgentState(BaseModel):
    """
    Represents the state of the Pitch Strength Agent's workflow.
    """
    pitch_id: str = Field(description="Unique identifier for the pitch being analyzed.")
    pitch_content: Optional[str] = Field(None, description="The raw text content of the pitch.")
    file_path: Optional[str] = Field(None, description="Original file path of the pitch (if uploaded).")
    
    # Analysis results from LLM
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Detailed LLM analysis of pitch components.")
    # Aggregated scores
    component_scores: Dict[str, int] = Field(default_factory=dict, description="Scores for each pitch component (0-100).")
    overall_score: Optional[int] = Field(None, description="Overall aggregated pitch score (0-100).")

    # Flags for privacy-preserving steps
    tee_processed: bool = Field(False, description="True if pitch was processed in a TEE.")
    zkp_hash: Optional[str] = Field(None, description="Hash of the Zero-Knowledge Proof for pitch integrity.")

    # On-chain transaction details
    on_chain_tx_hash: Optional[str] = Field(None, description="Transaction hash if pitch score was recorded on-chain.")
    
    error: Optional[str] = Field(None, description="Any error message encountered during processing.")


# --- 2. Initialize Tools / Services ---

# Initialize NLPProcessor (assumes it handles file reading internally or takes content)
nlp_processor_instance = NLPProcessor() # This instance will now correctly use the updated imports/config

# Initialize LLM and Embeddings Model with the updated model name
llm_for_agent = ChatGoogleGenerativeAI(model=gemini_llm_model, temperature=0.1, google_api_key=nlp_processor_instance.gemini_api_key)
embeddings_model_for_agent = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=nlp_processor_instance.gemini_api_key)


# Mock Ethereum interaction for PitchRegistry
class MockWeb3:
    def __init__(self, contract_address: str):
        self.contract_address = contract_address
        print(f"MockWeb3: Initialized for contract at {contract_address}")

    async def record_pitch_score(self, pitch_id_bytes: bytes, overall_score: int, zkp_hash_bytes: bytes):
        """Mocks sending a transaction to the PitchRegistry smart contract."""
        print(f"\n[MockWeb3] Simulating on-chain recording for pitch ID {pitch_id_bytes.hex()}...")
        print(f"  Overall Score: {overall_score}")
        print(f"  ZKP Hash: {zkp_hash_bytes.hex()}")
        await asyncio.sleep(1) # Simulate transaction time
        print(f"[MockWeb3] Pitch score recorded on-chain (mock).")
        return {"transaction_hash": f"0x{os.urandom(32).hex()}"} # Mock Tx hash

# Instantiate the mock Web3 client (replace with actual web3.py client)
MOCK_CONTRACT_ADDRESS = "0xPitchRegistryAddress" # Placeholder address
web3_client = MockWeb3(MOCK_CONTRACT_ADDRESS)


# --- 3. Define Agent Nodes (Functions) ---

async def ingest_pitch(state: PitchStrengthAgentState) -> PitchStrengthAgentState:
    """
    Ingests the pitch content from text or file, and handles initial loading.
    """
    print(f"\n[Pitch Strength Agent] Ingesting pitch for ID: {state.pitch_id}...")
    if not state.pitch_content and not state.file_path:
        return {"error": "No pitch content or file path provided."}

    if not state.pitch_content and state.file_path:
        return {"error": "File path provided but pitch_content is empty in state. Frontend should have pre-read the file."}
    
    print(f"[Pitch Strength Agent] Pitch content ingested (ready for analysis).")
    return {} # No direct state change here, pitch_content is already in state


async def analyze_pitch_in_tee(state: PitchStrengthAgentState) -> PitchStrengthAgentState:
    """
    Mocks pitch analysis within a Trusted Execution Environment (TEE).
    This node would handle the secure decryption and processing of sensitive pitch data.
    """
    print(f"\n[Pitch Strength Agent] Simulating TEE-secured pitch analysis for ID: {state.pitch_id}...")
    if state.error:
        print("[Pitch Strength Agent] Skipping TEE due to prior error.")
        return {}
    if not state.pitch_content:
        return {"error": "Pitch content missing for TEE analysis."}

    try:
        analysis = await nlp_processor_instance.analyze_pitch_content(
            state.pitch_content, llm_for_agent, embeddings_model_for_agent
        )
        
        await asyncio.sleep(0.7) # Simulate TEE processing time
        print(f"[Pitch Strength Agent] Pitch analysis (mock) conducted in TEE.")
        print(f"[Pitch Strength Agent] Raw analysis_results received from NLPProcessor: {json.dumps(analysis, indent=2)}")
        return {"tee_processed": True, "analysis_results": analysis}
    except Exception as e:
        print(f"Error during TEE analysis simulation: {e}")
        return {"error": f"TEE analysis simulation failed: {e}"}


async def calculate_scores(state: PitchStrengthAgentState) -> PitchStrengthAgentState:
    """
    Calculates component scores (0-100) and an overall score from the analysis results.
    Ensures all scores are properly extracted and scaled.
    """
    print(f"\n[Pitch Strength Agent] Calculating scores for ID: {state.pitch_id}...")
    if state.error or not state.analysis_results:
        print("[Pitch Strength Agent] Skipping score calculation due to prior error or missing analysis results.")
        # Return default scores if analysis results are missing
        return {
            "component_scores": {"clarity": 0, "team_strength": 0, "market_fit": 0, "originality": 0},
            "overall_score": 0,
            "error": state.error or "Analysis results missing."
        }

    try:
        component_scores = {}
        total_score_sum_for_avg = 0
        num_components_for_avg = 0 

        expected_components = ['clarity', 'team_strength', 'market_fit', 'originality']
        for component in expected_components:
            score_1_to_10 = state.analysis_results.get("components", {}).get(component, {}).get("score")
            
            if score_1_to_10 is not None and isinstance(score_1_to_10, (int, float)):
                scaled_score_0_to_100 = int(score_1_to_10 * 10) # Scale 1-10 to 10-100
                component_scores[component] = scaled_score_0_to_100
                total_score_sum_for_avg += scaled_score_0_to_100
                num_components_for_avg += 1
            else:
                component_scores[component] = 0 # Default to 0 if score is missing or invalid

        overall_score_from_nlp = state.analysis_results.get("overall_score")
        
        final_overall_score = 0
        if isinstance(overall_score_from_nlp, (int, float)):
            final_overall_score = int(overall_score_from_nlp) 
        elif num_components_for_avg > 0:
            final_overall_score = round(total_score_sum_for_avg / num_components_for_avg)
        
        print(f"[Pitch Strength Agent] Calculated component scores (0-100): {json.dumps(component_scores, indent=2)}")
        print(f"[Pitch Strength Agent] Final calculated overall score (0-100): {final_overall_score}")
        
        return {"component_scores": component_scores, "overall_score": final_overall_score}
    except Exception as e:
        print(f"Error calculating scores: {e}")
        return {"error": f"Error calculating scores: {e}"}


async def generate_zkp_for_pitch(state: PitchStrengthAgentState) -> PitchStrengthAgentState:
    """
    Mocks the generation of a Zero-Knowledge Proof for the integrity of the pitch analysis.
    This would prove that the overall_score was correctly derived from the (private) pitch data.
    """
    print(f"\n[Pitch Strength Agent] Simulating ZKP generation for pitch ID: {state.pitch_id}...")
    if state.error or state.overall_score is None: 
        print("[Pitch Strength Agent] Skipping ZKP generation due to prior error or missing overall score.")
        return {"error": state.error or "Overall score missing for ZKP generation."}

    try:
        proof_inputs = {
            "pitch_id": state.pitch_id,
            "overall_score": state.overall_score,
            "component_scores": state.component_scores 
        }
        zkp_hash = hashlib.sha256(json.dumps(proof_inputs, sort_keys=True).encode('utf-8')).hexdigest()
        
        await asyncio.sleep(0.5) # Simulate ZKP generation time
        print(f"[Pitch Strength Agent] ZKP (mock) generated with hash: {zkp_hash}")
        return {"zkp_hash": "0x" + zkp_hash}
    except Exception as e:
        print(f"Error generating ZKP: {e}")
        return {"error": f"ZKP generation simulation failed: {e}"}


async def record_score_on_chain(state: PitchStrengthAgentState) -> PitchStrengthAgentState:
    """
    Records the overall pitch score and ZKP hash on the blockchain.
    """
    print(f"\n[Pitch Strength Agent] Recording score on-chain for ID: {state.pitch_id}...")
    if state.error or state.overall_score is None or not state.zkp_hash: 
        print("[Pitch Strength Agent] Skipping on-chain record due to prior error or missing data.")
        return {"error": state.error or "Missing overall score or ZKP hash for on-chain record."}

    try:
        pitch_id_bytes = hashlib.sha256(state.pitch_id.encode('utf-8')).digest() 
        zkp_hash_bytes = bytes.fromhex(state.zkp_hash[2:]) 

        tx_receipt = await web3_client.record_pitch_score(
            pitch_id_bytes,
            state.overall_score,
            zkp_hash_bytes
        )
        print(f"[Pitch Strength Agent] On-chain record successful! Tx Hash: {tx_receipt['transaction_hash']}")
        return {"on_chain_tx_hash": tx_receipt['transaction_hash']}
    except Exception as e:
        print(f"Error recording score on-chain: {e}")
        return {"error": f"Failed to record score on-chain: {e}"}


async def final_output_pitch_strength(state: PitchStrengthAgentState) -> Dict[str, Any]:
    """
    Prepares the final output of the Pitch Strength Agent.
    Ensures all fields expected by the orchestrator are present and correctly typed.
    """
    print(f"\n[Pitch Strength Agent] Finalizing output for ID: {state.pitch_id}...")
    final_status = "completed"
    final_error = None
    if state.error:
        final_status = "failed"
        final_error = state.error
        print(f"[Pitch Strength Agent] Agent finished with error: {final_error}")

    final_data = {
        "pitch_id": state.pitch_id,
        "overall_score": state.overall_score if state.overall_score is not None else 0, # Default to 0 if None
        "component_scores": state.component_scores if state.component_scores is not None else {},
        "analysis_results": state.analysis_results if state.analysis_results is not None else {"components":{}}, 
        "privacy_flags": {
            "tee_processed": state.tee_processed,
            "zkp_hash": state.zkp_hash
        },
        "status": final_status, 
        "on_chain_tx_hash": state.on_chain_tx_hash
    }
    print(f"[Pitch Strength Agent] Agent final output for ID {state.pitch_id}: {json.dumps(final_data, indent=2)}")
    print(f"[Pitch Strength Agent] Agent completed with status: {final_status}.")
    return final_data


# --- 4. Build the LangGraph Workflow ---

workflow = StateGraph(PitchStrengthAgentState)

# Add nodes
workflow.add_node("ingest_pitch_node", ingest_pitch)
workflow.add_node("analyze_tee_node", analyze_pitch_in_tee)
workflow.add_node("calculate_scores_node", calculate_scores)
workflow.add_node("generate_zkp_node", generate_zkp_for_pitch)
workflow.add_node("record_on_chain_node", record_score_on_chain)
workflow.add_node("final_output_node", final_output_pitch_strength)

# Define entry point
workflow.set_entry_point("ingest_pitch_node")

# Define edges (transitions)
workflow.add_edge("ingest_pitch_node", "analyze_tee_node")
workflow.add_edge("analyze_tee_node", "calculate_scores_node")
workflow.add_edge("calculate_scores_node", "generate_zkp_node")
workflow.add_edge("generate_zkp_node", "record_on_chain_node")
workflow.add_edge("record_on_chain_node", "final_output_node")
workflow.add_edge("final_output_node", END)

# Compile the graph
pitch_strength_app = workflow.compile()


# --- 5. Example Usage (for direct testing) ---
async def run_pitch_strength_agent(pitch_id: str, pitch_content: Optional[str] = None, file_path: Optional[str] = None):
    """
    Function to run the Pitch Strength Agent workflow.
    """
    initial_state = PitchStrengthAgentState(
        pitch_id=pitch_id,
        pitch_content=pitch_content,
        file_path=file_path
    )
    print(f"\n--- Starting Pitch Strength Agent for Pitch ID: {pitch_id} ---")
    final_output = None
    try:
        final_output = await pitch_strength_app.ainvoke(initial_state, {"recursion_limit": 100})
    except Exception as e:
        print(f"Error during Pitch Strength Agent execution: {e}")
        final_output = {"status": "failed", "pitch_id": pitch_id, "error": str(e)}

    print(f"--- Finished Pitch Strength Agent for Pitch ID: {pitch_id} ---")
    return final_output

if __name__ == "__main__":
    dummy_pitch = """
    Our startup, Quantum Leap Solutions, aims to revolutionize the supply chain industry using a novel blend of AI and quantum computing.
    The team consists of Dr. Alice Smith, a leading expert in quantum algorithms, and Mr. Bob Johnson, a seasoned supply chain veteran with 15 years of experience at global logistics firms.
    We identified a critical bottleneck in real-time inventory management, causing billions in losses annually. Our solution provides predictive analytics with unparalleled accuracy, reducing waste by 30%.
    The market for supply chain optimization is growing rapidly, with a projected value of $50 billion by 2030. We have secured early partnerships with two major manufacturing companies.
    """
    
    # Example 1: Run with pitch content
    asyncio.run(run_pitch_strength_agent("pitch_001", pitch_content=dummy_pitch))

