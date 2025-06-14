import asyncio
import json
import os
import hashlib # For ZKP hash mock and graph hash
from typing import Dict, Any, List, Literal, Optional

# Correct Pydantic import
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

# Correct relative import for GraphProcessor (assuming it's in the same folder or an accessible sibling)
from .graph_processor import GraphProcessor # Assumes graph_processor.py is directly in agent_social_trust/

# --- 1. Define Agent State ---
class SocialGraphAgentState(BaseModel):
    """
    Represents the state of the Social Trust Graph Agent's workflow.
    """
    # ID for the batch/session of graph updates
    session_id: str = Field(description="Unique identifier for the current graph update session.")
    # Raw data for graph updates (e.g., new endorsements, collaborations, team formations)
    new_graph_data: List[Dict[str, Any]] = Field(default_factory=list, description="List of new data points to add to the graph.")
    # Current trust scores derived from the graph
    current_trust_scores: Dict[str, int] = Field(default_factory=dict, description="Calculated trust scores for entities.") # Initialize as dict
    # Hash of the current graph state for on-chain anchoring
    current_graph_hash: Optional[str] = Field(None, description="Cryptographic hash of the current graph state.") # Allow None
    # Flag for ZKP of endorsement (conceptual)
    zk_verified_endorsements_processed: bool = Field(False, description="Flag indicating if ZK-verified endorsements were conceptually processed.")
    # TEE processing for sensitive graph analysis (conceptual)
    tee_analysis_conducted: bool = Field(False, description="Flag indicating if sensitive graph analysis was conducted in TEE.")
    # On-chain transaction details
    on_chain_tx_hash: Optional[str] = Field(None, description="Transaction hash if graph hash was recorded on-chain.")
    error: Optional[str] = Field(None, description="Any error message encountered during processing.") # Allow None


# --- 2. Initialize Tools / Services ---
# Instantiate the GraphProcessor. In a real system, this manages a persistent graph DB.
graph_processor_instance = GraphProcessor()

# Mock Ethereum interaction for TrustGraphRegistry
class MockTrustWeb3:
    def __init__(self, contract_address: str):
        self.contract_address = contract_address
        print(f"MockTrustWeb3: Initialized for contract at {contract_address}")

    async def record_graph_hash(self, session_id_bytes: bytes, graph_hash_bytes: bytes):
        """Mocks sending a transaction to the TrustGraphRegistry smart contract."""
        print(f"\n[MockTrustWeb3] Simulating on-chain recording for session {session_id_bytes.hex()}...")
        print(f"  Graph Hash: {graph_hash_bytes.hex()}")
        await asyncio.sleep(1) # Simulate transaction time
        print(f"[MockTrustWeb3] Graph hash recorded on-chain (mock).")
        return {"transaction_hash": f"0x{os.urandom(32).hex()}"} # Mock Tx hash

# Instantiate the mock Web3 client (replace with actual web3.py client)
MOCK_TRUST_CONTRACT_ADDRESS = "0xTrustGraphRegistryAddress" # Placeholder address
trust_web3_client = MockTrustWeb3(MOCK_TRUST_CONTRACT_ADDRESS)


# --- 3. Define Agent Nodes (Functions) ---

async def ingest_new_graph_data(state: SocialGraphAgentState) -> SocialGraphAgentState:
    """
    Ingests new data points (collaborations, endorsements, team formations) into the graph.
    This node processes a batch of updates.
    """
    print(f"\n[Social Trust Graph Agent] Ingesting new data for session: {state.session_id}...")
    if not state.new_graph_data:
        print("[Social Trust Graph Agent] No new data to ingest. Skipping ingestion.")
        # If no new data, it's not an error but a skip for this step
        return {"error": None} # Ensure error is explicitly None if no error

    ingestion_error = None
    for item in state.new_graph_data:
        try:
            item_type = item.get("type")
            if item_type == "collaboration":
                await graph_processor_instance.add_collaboration(
                    item["entity1_id"], item["entity2_id"], item.get("weight", 1.0)
                )
            elif item_type == "endorsement":
                await graph_processor_instance.add_endorsement(
                    item["endorsed_id"], item["endorser_id"], item.get("strength", 1.0)
                )
            elif item_type == "team_formation":
                await graph_processor_instance.add_team_formation(
                    item["members_ids"], item["team_id"], item.get("shared_experience_weight", 1.0)
                )
            else:
                print(f"[Social Trust Graph Agent] Unknown data type: {item_type} for item {item}. Skipping.")
        except KeyError as e:
            ingestion_error = f"Missing key in graph data item: {e} for item {item}"
            print(f"[Social Trust Graph Agent] Ingestion Error: {ingestion_error}")
            break # Stop processing further items if a key error occurs
        except Exception as e:
            ingestion_error = f"Error ingesting graph data item {item}: {e}"
            print(f"[Social Trust Graph Agent] Ingestion Error: {ingestion_error}")
            break

    return {"error": ingestion_error} # Return error if any occurred, else None

async def analyze_graph_in_tee(state: SocialGraphAgentState) -> SocialGraphAgentState:
    """
    Mocks sensitive graph analysis within a Trusted Execution Environment (TEE).
    """
    print(f"\n[Social Trust Graph Agent] Simulating TEE-secured graph analysis for session: {state.session_id}...")
    if state.error:
        print("[Social Trust Graph Agent] Skipping TEE dueable to prior error.")
        return {} # No state change if skipping due to error
    
    try:
        await asyncio.sleep(0.7) # Simulate TEE processing time
        print(f"[Social Trust Graph Agent] Sensitive graph analysis (mock) conducted in TEE.")
        return {"tee_analysis_conducted": True}
    except Exception as e:
        print(f"Error during TEE analysis simulation: {e}")
        return {"error": f"TEE analysis simulation failed: {e}"}

async def calculate_and_hash_trust(state: SocialGraphAgentState) -> SocialGraphAgentState:
    """
    Calculates trust scores and generates a hash of the current graph state.
    """
    print(f"\n[Social Trust Graph Agent] Calculating trust scores and hashing graph for session: {state.session_id}...")
    if state.error:
        print("[Social Trust Graph Agent] Skipping trust calculation due to prior error.")
        return {} # No state change if skipping due to error
    
    try:
        trust_scores = await graph_processor_instance.calculate_trust_scores()
        graph_hash = await graph_processor_instance.get_graph_hash()
        
        print(f"[Social Trust Graph Agent] Calculated trust scores: {json.dumps(trust_scores, indent=2)}")
        print(f"[Social Trust Graph Agent] Generated graph hash: {graph_hash}")
        
        return {
            "current_trust_scores": trust_scores,
            "current_graph_hash": graph_hash
        }
    except Exception as e:
        print(f"Error calculating trust or hashing graph: {e}")
        return {"error": f"Error calculating trust or hashing graph: {e}"}

async def generate_zkp_for_endorsement(state: SocialGraphAgentState) -> SocialGraphAgentState:
    """
    Mocks the generation of Zero-Knowledge Proofs for *individual endorsements*.
    """
    print(f"\n[Social Trust Graph Agent] Simulating ZKP generation for relevant endorsements in session: {state.session_id}...")
    if state.error:
        print("[Social Trust Graph Agent] Skipping ZKP generation due to prior error.")
        return {} # No state change if skipping due to error
    if not state.new_graph_data:
        print("[Social Trust Graph Agent] No new data for ZKP verification. Skipping.")
        return {} # No state change, not an error

    zk_verified = False
    for item in state.new_graph_data:
        if item.get("type") == "endorsement":
            mock_endorsement_proof_inputs = {
                "endorser": item["endorser_id"],
                "endorsed": item["endorsed_id"],
                "strength": item.get("strength", 1.0)
            }
            mock_zkp_hash = hashlib.sha256(json.dumps(mock_endorsement_proof_inputs, sort_keys=True).encode()).hexdigest()
            print(f"  Mock ZKP generated for endorsement from {item['endorser_id']} to {item['endorsed_id']}: {mock_zkp_hash}")
            zk_verified = True 
            # In a real system, you'd verify this ZKP and potentially store its hash or validity status.
            
    if zk_verified:
        print(f"[Social Trust Graph Agent] ZK-verified endorsements (mock) processed for session {state.session_id}.")
        return {"zk_verified_endorsements_processed": True}
    else:
        return {"zk_verified_endorsements_processed": False} # Explicitly false if no endorsements found

async def record_graph_hash_on_chain(state: SocialGraphAgentState) -> SocialGraphAgentState:
    """
    Records the current graph hash on the blockchain.
    """
    print(f"\n[Social Trust Graph Agent] Recording graph hash on-chain for session: {state.session_id}...")
    if state.error or not state.current_graph_hash:
        print("[Social Trust Graph Agent] Skipping on-chain record due to prior error or missing graph hash.")
        return {"error": state.error or "Missing graph hash for on-chain record."}

    try:
        session_id_bytes = hashlib.sha256(state.session_id.encode('utf-8')).digest()
        graph_hash_bytes = bytes.fromhex(state.current_graph_hash[2:]) 

        tx_receipt = await trust_web3_client.record_graph_hash(
            session_id_bytes,
            graph_hash_bytes
        )
        print(f"[Social Trust Graph Agent] On-chain record successful! Tx Hash: {tx_receipt['transaction_hash']}")
        return {"on_chain_tx_hash": tx_receipt['transaction_hash']}
    except Exception as e:
        print(f"Error recording graph hash on-chain: {e}")
        return {"error": f"Failed to record graph hash on-chain: {e}"}


async def final_output_social_graph(state: SocialGraphAgentState) -> Dict[str, Any]:
    """
    Prepares the final output of the Social Trust Graph Agent.
    Ensures a consistent structure with defaults for the orchestrator.
    """
    print(f"\n[Social Trust Graph Agent] Finalizing output for session: {state.session_id}...")
    final_status = "completed"
    final_error = None
    if state.error:
        final_status = "failed"
        final_error = state.error
        print(f"[Social Trust Graph Agent] Agent finished with error: {final_error}")
    
    final_data = {
        "session_id": state.session_id,
        "overall_trust_scores": state.current_trust_scores if state.current_trust_scores is not None else {},
        "current_graph_hash": state.current_graph_hash,
        "privacy_flags": {
            "zk_verified_endorsements_processed": state.zk_verified_endorsements_processed,
            "tee_analysis_conducted": state.tee_analysis_conducted
        },
        "status": final_status, # Explicitly set final status
        "on_chain_tx_hash": state.on_chain_tx_hash
    }
    print(f"[Social Trust Graph Agent] Agent final output for session {state.session_id}: {json.dumps(final_data, indent=2)}")
    return final_data


# --- 4. Build the LangGraph Workflow ---

workflow = StateGraph(SocialGraphAgentState)

# Add nodes
workflow.add_node("ingest_data", ingest_new_graph_data)
workflow.add_node("process_zkp_endorsement", generate_zkp_for_endorsement)
workflow.add_node("analyze_tee_graph", analyze_graph_in_tee)
workflow.add_node("calculate_hash_trust", calculate_and_hash_trust)
workflow.add_node("record_graph_on_chain", record_graph_hash_on_chain)
workflow.add_node("final_output", final_output_social_graph)

# Define entry point
workflow.set_entry_point("ingest_data")

# Define edges (transitions)
workflow.add_edge("ingest_data", "process_zkp_endorsement") 
workflow.add_edge("process_zkp_endorsement", "analyze_tee_graph") 
workflow.add_edge("analyze_tee_graph", "calculate_hash_trust") 
workflow.add_edge("calculate_hash_trust", "record_graph_on_chain") 
workflow.add_edge("record_graph_on_chain", "final_output") 
workflow.add_edge("final_output", END)

# Compile the graph
social_trust_graph_app = workflow.compile()


# --- 5. Example Usage (for direct testing) ---
async def run_social_trust_graph_agent(session_id: str, new_graph_data: List[Dict[str, Any]]) -> Dict[str, Any]: # Explicitly type return
    """
    Function to run the Social Trust Graph Agent workflow.
    Uses ainvoke to return final state directly.
    """
    initial_state = SocialGraphAgentState(
        session_id=session_id,
        new_graph_data=new_graph_data
    )
    print(f"\n--- Starting Social Trust Graph Agent for Session ID: {session_id} ---")
    final_output = None
    try:
        # --- CRITICAL FIX: Use ainvoke to get final state directly ---
        final_output = await social_trust_graph_app.ainvoke(initial_state, {"recursion_limit": 100})
    except Exception as e:
        print(f"Error during Social Trust Graph Agent execution: {e}")
        # Return a failed status with error if an exception occurs
        final_output = {"status": "failed", "session_id": session_id, "error": str(e)}

    print(f"--- Finished Social Trust Graph Agent for Session ID: {session_id} ---")
    return final_output

if __name__ == "__main__":
    mock_new_data_batch = [
        {"type": "collaboration", "entity1_id": "founder_A", "entity2_id": "contributor_X", "weight": 0.8},
        {"type": "endorsement", "endorser_id": "investor_V", "endorsed_id": "founder_A", "strength": 0.9},
    ]
    asyncio.run(run_social_trust_graph_agent("graph_update_session_001", mock_new_data_batch))

    print("\n\n--- Running Agent with No New Data ---")
    asyncio.run(run_social_trust_graph_agent("graph_update_session_002", []))
