import networkx as nx
import hashlib
import json
from typing import Dict, List, Any, Tuple
import asyncio
import random

class GraphProcessor:
    def __init__(self):
        """
        Initializes the GraphProcessor.
        In a real system, this might connect to a persistent graph database.
        """
        self.trust_graph = nx.DiGraph() # Directed graph for trust relationships (e.g., A endorses B)
        print("GraphProcessor: Initialized empty trust graph.")

    def _add_node_if_not_exists(self, node_id: str):
        """Helper to add a node to the graph if it doesn't already exist."""
        if not self.trust_graph.has_node(node_id):
            self.trust_graph.add_node(node_id)

    async def add_collaboration(self, entity1_id: str, entity2_id: str, weight: float = 1.0):
        """
        Adds a collaboration edge between two entities.
        Weight can represent intensity or duration of collaboration.
        Simulates on-chain collaboration events.
        """
        self._add_node_if_not_exists(entity1_id)
        self._add_node_if_not_exists(entity2_id)
        # Use a higher weight for collaboration in both directions if mutual
        self.trust_graph.add_edge(entity1_id, entity2_id, type='collaboration', weight=weight)
        self.trust_graph.add_edge(entity2_id, entity1_id, type='collaboration', weight=weight)
        print(f"GraphProcessor: Added collaboration between {entity1_id} and {entity2_id} with weight {weight}.")
        await asyncio.sleep(0.05) # Simulate processing time

    async def add_endorsement(self, endorser_id: str, endorsed_id: str, strength: float = 1.0):
        """
        Adds a directed endorsement edge from endorser to endorsed.
        Strength represents the value/trust of the endorsement.
        Simulates ZK-verified endorsement events.
        """
        self._add_node_if_not_exists(endorser_id)
        self._add_node_if_not_exists(endorsed_id)
        # Endorsements are typically directed
        self.trust_graph.add_edge(endorser_id, endorsed_id, type='endorsement', weight=strength)
        print(f"GraphProcessor: Added endorsement from {endorser_id} to {endorsed_id} with strength {strength}.")
        await asyncio.sleep(0.05) # Simulate processing time

    async def add_team_formation(self, members_ids: List[str], team_id: str, shared_experience_weight: float = 1.0):
        """
        Adds edges representing a team formation, connecting team members to a central team node.
        Also adds collaboration edges between team members.
        """
        self._add_node_if_not_exists(team_id)
        for member_id in members_ids:
            self._add_node_if_not_exists(member_id)
            self.trust_graph.add_edge(member_id, team_id, type='team_member_of', weight=1.0)
            self.trust_graph.add_edge(team_id, member_id, type='team_has_member', weight=1.0)
            # Add implicit collaborations between team members
            for other_member_id in members_ids:
                if member_id != other_member_id:
                    self.trust_graph.add_edge(member_id, other_member_id, type='team_collaboration', weight=shared_experience_weight)
        print(f"GraphProcessor: Added team formation for {team_id} with members {members_ids}.")
        await asyncio.sleep(0.05) # Simulate processing time


    async def calculate_trust_scores(self) -> Dict[str, float]:
        """
        Calculates a dynamic trust score for each node (entity) in the graph.
        Uses a combination of PageRank for influence and weighted degree for activity.
        """
        if not self.trust_graph.nodes:
            print("GraphProcessor: No nodes in graph to calculate trust scores.")
            return {}

        print("GraphProcessor: Calculating trust scores using PageRank and weighted degree...")
        # 1. PageRank for influence
        # We can adjust alpha (damping factor) based on how much 'influence' propagates
        # Max_iter and tol for convergence
        try:
            # Consider only 'endorsement' edges for a specific PageRank calculation
            # Or use a generic PageRank on the entire graph, weights based on edge 'weight' attribute
            pagerank_scores = nx.pagerank(self.trust_graph, alpha=0.85, weight='weight')
        except nx.PowerIterationFailedConvergence:
            print("GraphProcessor: PageRank failed to converge, using default values.")
            pagerank_scores = {node: 0.1 for node in self.trust_graph.nodes}
        except Exception as e:
            print(f"GraphProcessor: Error in PageRank calculation: {e}. Assigning default scores.")
            pagerank_scores = {node: random.random() * 0.2 for node in self.trust_graph.nodes} # Assign small random scores

        # 2. Weighted Degree for activity/connectedness
        # Sum of weights of all incoming and outgoing edges
        weighted_degree_scores = {
            node: sum(self.trust_graph[node][neighbor]['weight'] for neighbor in self.trust_graph.successors(node) if 'weight' in self.trust_graph[node][neighbor]) +
                  sum(self.trust_graph[neighbor][node]['weight'] for neighbor in self.trust_graph.predecessors(node) if 'weight' in self.trust_graph[neighbor][node])
            for node in self.trust_graph.nodes
        }

        # Normalize weighted degree scores to a 0-1 range for blending
        max_weighted_degree = max(weighted_degree_scores.values()) if weighted_degree_scores else 0
        normalized_weighted_degree = {
            node: score / max_weighted_degree if max_weighted_degree > 0 else 0
            for node, score in weighted_degree_scores.items()
        }

        # Combine PageRank and Weighted Degree (simple average for now, can be weighted)
        trust_scores = {}
        for node in self.trust_graph.nodes:
            # We want PageRank to reflect influence, and weighted degree to reflect activity.
            # Blend them, then scale to 0-100 (or 0-1 for later use)
            combined_score = (pagerank_scores.get(node, 0) + normalized_weighted_degree.get(node, 0)) / 2
            # Scale to 0-100 for a more intuitive score
            trust_scores[node] = round(combined_score * 100)

        print(f"GraphProcessor: Calculated trust scores for {len(trust_scores)} entities.")
        await asyncio.sleep(0.5)
        return trust_scores

    async def get_graph_hash(self) -> str:
        """
        Generates a cryptographic hash of the current state of the trust graph.
        This hash can be stored on-chain for integrity verification.
        """
        if not self.trust_graph.nodes:
            return "0x" + hashlib.sha256(b"empty_graph").hexdigest()

        # Deterministically serialize the graph for consistent hashing
        # Sorting nodes and edges is crucial for consistent hashing
        nodes_sorted = sorted(self.trust_graph.nodes())
        edges_list = []
        for u, v, data in self.trust_graph.edges(data=True):
            # Sort attributes within edge data to ensure consistency
            sorted_data = json.dumps(dict(sorted(data.items())))
            edges_list.append((u, v, sorted_data))
        edges_sorted = sorted(edges_list) # Sort edges themselves

        graph_data = {
            "nodes": nodes_sorted,
            "edges": edges_sorted
        }
        graph_json = json.dumps(graph_data, sort_keys=True) # Ensure overall sorting
        graph_hash = hashlib.sha256(graph_json.encode('utf-8')).hexdigest()
        print(f"GraphProcessor: Generated graph hash: {graph_hash}")
        return "0x" + graph_hash

# Example Usage (for testing GraphProcessor directly)
async def test_graph_processor():
    processor = GraphProcessor()

    print("\n--- Adding mock data ---")
    await processor.add_collaboration("founder_A", "contributor_X", weight=0.8)
    await processor.add_collaboration("founder_B", "contributor_X", weight=0.5)
    await processor.add_endorsement("investor_V", "founder_A", strength=0.9)
    await processor.add_endorsement("contributor_Y", "founder_A", strength=0.7)
    await processor.add_endorsement("founder_A", "contributor_X", strength=0.6) # Founder endorsing contributor
    await processor.add_team_formation(["founder_A", "dev_1", "dev_2"], "team_Alpha", shared_experience_weight=0.9)
    await processor.add_collaboration("founder_A", "investor_V", weight=0.2) # Investor-founder interaction

    print("\n--- Calculating trust scores ---")
    trust_scores = await processor.calculate_trust_scores()
    print(json.dumps(trust_scores, indent=2))

    print("\n--- Generating graph hash ---")
    current_graph_hash = await processor.get_graph_hash()
    print(f"Current Graph Hash: {current_graph_hash}")

    print("\n--- Adding more data and re-calculating ---")
    await processor.add_endorsement("investor_W", "founder_B", strength=0.95)
    await processor.add_collaboration("founder_C", "dev_1", weight=0.7)
    new_trust_scores = await processor.calculate_trust_scores()
    print(json.dumps(new_trust_scores, indent=2))

    new_graph_hash = await processor.get_graph_hash()
    print(f"New Graph Hash: {new_graph_hash}")
    print(f"Hashes are different: {current_graph_hash != new_graph_hash}")


if __name__ == "__main__":
    asyncio.run(test_graph_processor())
