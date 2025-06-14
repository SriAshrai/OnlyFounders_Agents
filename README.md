OnlyFounders AI Agents: Decentralized Pitch & Trust Evaluation System

🚀 Project Overview


The OnlyFounders AI Agents project introduces a cutting-edge, decentralized system designed to revolutionize the way startup pitches are evaluated and how trust is established within entrepreneurial ecosystems. Leveraging the power of advanced AI models (Gemini), LangGraph for orchestrating complex agentic workflows, and Web3 principles for transparent and verifiable data, this system provides a robust framework for assessing pitch strength and building a social trust graph among founders and collaborators.

Our system addresses critical challenges in early-stage venture assessment:

Subjectivity in Pitch Evaluation: By employing AI, we aim for consistent, data-driven analysis.
Lack of Verifiable Trust: Integrating Web3 (mock) components allows for immutable records and potential future on-chain trust mechanisms.
Scalability of Analysis: Agent-based architecture enables modularity and concurrent processing.
This project serves as a foundational proof-of-concept for a future where AI-driven insights meet decentralized verification, fostering a more equitable and efficient startup landscape.

✨ Key Features


Intelligent Pitch Strength Agent: Utilizes Google's Gemini LLM to analyze startup pitches across critical dimensions:
Clarity & Structure: Evaluates the coherence and presentation of the pitch.
Team Strength: Assesses the credibility and expertise of the founding team.
Market Fit: Determines the alignment of the solution with market needs and opportunities.
Originality: Benchmarks the pitch against existing concepts to identify novelty.
Decentralized Social Trust Graph Agent: Builds and manages a conceptual social trust network based on interactions like collaborations, endorsements, and team formations.
Dynamic Trust Scoring: Calculates trust scores for entities within the network.
Graph Hashing: Generates cryptographic hashes of the graph state for potential on-chain anchoring.
LangGraph Orchestration: A sophisticated orchestrator manages the flow between different AI agents, enabling complex multi-step workflows.
Mock Web3 Integration: Simulates on-chain recording of pitch scores and trust graph hashes, demonstrating the potential for verifiable and immutable data.
Privacy-Preserving Concepts: Includes conceptual implementations for:
Trusted Execution Environments (TEEs): For secure and private AI analysis of sensitive pitch data.
Zero-Knowledge Proofs (ZKPs): For verifiable claims (e.g., pitch score validity, endorsement integrity) without revealing underlying sensitive information.
Scalable & Modular Architecture: Built with independent agents, allowing for easy expansion and integration of new functionalities.
⚙️ Architecture & Flow
The OnlyFounders AI Agents system is designed with a clear separation of concerns, orchestrated by a central component:

Code snippet

graph TD
    A[User/Frontend Request] --> B(Flask Backend - app.py);
    B --> C(Main Orchestrator - main_orchestrator.py);

    subgraph Agentic Workflow
        C --1. Calls with Pitch Data--> D(Pitch Strength Agent);
        D --> D1{Ingest Pitch};
        D1 --> D2{TEE Analysis Mock};
        D2 --> D3{Calculate Scores};
        D3 --> D4{ZKP Generation Mock};
        D4 --> D5{Mock On-Chain Record};
        D5 --> C1(Return PS Results to Orchestrator);
    end

    C --2. Prepares STG Input--> E(Social Trust Graph Agent);
    E --> E1{Ingest Graph Data};
    E1 --> E2{ZKP Endorsement Mock};
    E2 --> E3{TEE Graph Analysis Mock};
    E3 --> E4{Calculate & Hash Trust};
    E4 --> E5{Mock On-Chain Record};
    E5 --> C2(Return STG Results to Orchestrator);

    C1 & C2 --> F(Final Orchestrator Output);
    F --> G(Flask Backend Response);
    G --> H[User/Frontend];

    style D fill:#f9f,stroke:#333,stroke-width:2px;
    style E fill:#ccf,stroke:#333,stroke-width:2px;
Workflow Breakdown:

User/Frontend Interaction: A user initiates a request (e.g., submits a pitch) to the Flask backend.
Flask Backend (app.py): Receives the request and acts as the entry point, passing the data to the Main Orchestrator.
Main Orchestrator (main_orchestrator.py):
Initializes the overall session and manages the sequential (or potentially parallel) execution of specialized AI agents.
Calls Pitch Strength Agent: Delegates the pitch analysis task.
Prepares STG Input: Based on the Pitch Strength Agent's output (e.g., overall score, identified founders), it prepares relevant data for the Social Trust Graph Agent. This signifies a crucial inter-agent communication step.
Calls Social Trust Graph Agent: Delegates the task of updating and analyzing the trust graph.
Aggregates Results: Collects outputs from all agents and compiles a comprehensive final response.
Pitch Strength Agent (agent_pitch_strength/Pitch_Strength_Langgraph.py):
Ingest Pitch: Loads pitch content (from direct text or simulated file input).
TEE Analysis (Mock): Simulates a secure environment for confidential AI processing.
Calculate Scores: Leverages the NLPProcessor (which uses Google's Gemini LLMs) to assess clarity, team, market fit, and originality.
ZKP Generation (Mock): Simulates creating a proof of analysis integrity.
Mock On-Chain Record: Uses MockWeb3 to simulate recording the pitch score and ZKP hash on a blockchain.
Social Trust Graph Agent (agent_social_trust/trust_graph_agent.py):
Ingest Graph Data: Processes new data points like collaborations or endorsements using the GraphProcessor.
ZKP Endorsement (Mock): Simulates ZKP generation for individual endorsements.
TEE Graph Analysis (Mock): Simulates secure analysis of sensitive graph data.
Calculate & Hash Trust: Updates trust scores within the graph and generates a hash of the current graph state.
Mock On-Chain Record: Uses MockTrustWeb3 to simulate recording the graph hash on a blockchain.
Return to Flask: The final, aggregated results from the orchestrator are sent back to the Flask application.
Frontend/User: The Flask backend delivers the response to the client.
Core Components & Technologies:

Python 3.9+: The primary programming language.
Flask: Lightweight web framework for the backend API.
Google Gemini API: Powers the large language model capabilities for pitch analysis (gemini-1.5-flash model) and embeddings (models/text-embedding-004).
LangChain / LangGraph: Frameworks for building and orchestrating complex AI agent workflows, managing state, and defining transitions.
python-dotenv: For securely loading environment variables (like your Gemini API key).
sentence-transformers: For efficient sentence embeddings, used in calculating pitch originality.
pypdf, python-docx: For text extraction from common document formats (conceptual, as the current system takes pre-extracted text).
web3.py (Mocked): Demonstrates interaction with blockchain smart contracts.
Pydantic: For robust data validation and state management within LangGraph.
🛠️ Setup & Installation
Follow these steps to get the OnlyFounders AI Agents running on your local machine.

Prerequisites
Python 3.9 or higher
pip (Python package installer)
1. Clone the Repository
Bash

git clone https://github.com/your-username/onlyfounders-ai-agents.git
cd onlyfounders-ai-agents
(Replace your-username with your actual GitHub username once you set up the repository.)

2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

Bash

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
(Create a requirements.txt file in your root directory if you haven't already. You can generate it using pip freeze > requirements.txt. Ensure it includes: Flask, python-dotenv, google-generativeai, langchain-google-genai, langchain-core, langgraph, pydantic, sentence-transformers, numpy, scikit-learn, pypdf, python-docx)

4. Configure Your Google Gemini API Key
Go to the Google AI Studio and generate a new API key.

In the root directory of your project (the onlyfounders-ai-agents folder), create a file named .env (if it doesn't already exist).

Add your API key to the .env file in the following format:

GEMINI_API_KEY="YOUR_GENERATED_GEMINI_API_KEY_HERE"
Important: Ensure there are no spaces around the = sign, and your key is enclosed in double-quotes.

5. Run the Flask Application
Once your dependencies are installed and your API key is configured, start the Flask server:

Bash

python app.py
You should see output indicating that the Flask app is running, typically on http://127.0.0.1:5000 (localhost).



🚀 Usage


With the Flask server running, you can interact with the AI agents via the exposed API endpoint.

API Endpoint
URL: /analyze-agents
Method: POST
Content-Type: application/json
Request Body
The request body should be a JSON object with the following structure:

JSON

{
  "session_id": "unique_session_identifier_e.g._session_001",
  "pitch_content": "Your startup pitch text goes here. This can be a few paragraphs.",
  "file_path": null
}
session_id: A unique string to identify your analysis session.
pitch_content: The full text of the startup pitch you want to analyze.
file_path: (Currently conceptual, should be null as the backend expects pitch_content directly). In a production environment, this would point to an uploaded file.
Example Request (using curl)
Open a new terminal window (keep your Flask server running in the first one) and execute:

Bash

curl -X POST -H "Content-Type: application/json" \
     -d '{
           "session_id": "my_first_pitch_analysis",
           "pitch_content": "Our new project, \"Decentralized Governance Protocol (DGP),\" aims to empower DAOs with truly liquid democracy. The team includes Dr. Ava Li (PhD in distributed systems, ex-ConsenSys) and Mr. Ben Carter (10 years in community management for large open-source projects). We identified a severe problem with voter apathy and whale dominance in current DAO structures. Our solution uses on-chain delegation and reputation-weighted voting to increase participation by 40% and fairness by 20%. The market for DAO tooling and governance solutions is rapidly maturing, projected to reach $10 billion by 2028. We have early commitments from three major DeFi protocols for pilot programs."
         }' \
     http://127.0.0.1:5000/analyze-agents
Expected Response
The API will return a JSON object containing the comprehensive analysis from both the Pitch Strength Agent and the Social Trust Graph Agent, along with overall status and any errors.

JSON

{
  "status": "completed",
  "session_id": "my_first_pitch_analysis",
  "error": null,
  "pitch_analysis": {
    "pitch_id": "pitch_my_first_pitch_analysis",
    "pitch_content": "...",
    "file_path": null,
    "analysis_results": {
      "overall_score": 7,
      "components": {
        "clarity": {
          "score": 7,
          "reasoning": "..."
        },
        "team_strength": {
          "score": 8,
          "reasoning": "..."
        },
        "market_fit": {
          "score": 6,
          "reasoning": "..."
        },
        "originality": {
          "score": 8,
          "reasoning": "..."
        }
      }
    },
    "component_scores": {
      "clarity": 70,
      "team_strength": 80,
      "market_fit": 60,
      "originality": 80
    },
    "overall_score": 73,
    "tee_processed": true,
    "zkp_hash": "0x...",
    "on_chain_tx_hash": "0x..."
  },
  "social_trust_analysis": {
    "session_id": "stg_session_my_first_pitch_analysis",
    "overall_trust_scores": {
      "founder_analysis": 0.73,
      "project_pitch_my_first_pitch_analysis": 0.73
    },
    "current_graph_hash": "0x...",
    "privacy_flags": {
      "zk_verified_endorsements_processed": true,
      "tee_analysis_conducted": true
    },
    "status": "completed",
    "on_chain_tx_hash": "0x..."
  }
}
(Note: Actual scores and reasoning will vary based on the LLM's output and your pitch content.)



📁 Project Structure


onlyfounders-ai-agents/
├── .env                  # Environment variables (GEMINI_API_KEY)
├── app.py                # Main Flask application entry point
├── main_orchestrator.py  # Orchestrates the multi-agent workflow
├── requirements.txt      # Python dependencies
├── agent_pitch_strength/
│   ├── __init__.py
│   ├── initial_setup.py  # Initializes NLPProcessor, LLM, and Embeddings
│   └── Pitch_Strength_Langgraph.py # Pitch Strength Agent workflow (LangGraph)
└── agent_social_trust/
    ├── __init__.py
    ├── graph_processor.py    # Manages the in-memory trust graph (nodes, edges)
    └── trust_graph_agent.py  # Social Trust Graph Agent workflow (LangGraph)

    
🧑‍💻 Contributing


We welcome contributions to enhance the OnlyFounders AI Agents project! If you're interested in contributing:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature-name).
Make your changes.
Commit your changes (git commit -m 'feat: Add new feature X').
Push to the branch (git push origin feature/your-feature-name).
Open a Pull Request.
Please ensure your code adheres to good practices, includes comments where necessary, and passes all tests (if applicable).


🙏 Acknowledgements
Google Gemini API: For powerful and versatile language models.
LangChain & LangGraph: For providing an excellent framework for building and orchestrating AI agents.
Python community: For the rich ecosystem of libraries and tools.

