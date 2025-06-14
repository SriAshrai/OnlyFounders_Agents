Building OnlyFounders AI Agents: My Vision for Decentralized Fundraising Excellence
🚀 Project Vision



I am actively building the world's first AI-native onchain fundraising engine for OnlyFounders. My core mission is to replace the outdated, pitch-based capital allocation model with a system rooted in verifiable, decentralized reputation. By skillfully leveraging advanced AI and swarm intelligence, I aim to create a protocol where access to capital is determined by proven reputation, not by connections or persuasive rhetoric.



✨ Core Philosophy



My approach is driven by the conviction that access to capital must be earned through proof, not persuasion. I am meticulously crafting a system where AI agents, operating with privacy-preserving capabilities, filter and score founders, investors, and partner contributors. These agents are trained on a rich tapestry of onchain data, intricate trust graphs, and securely encrypted narratives, ensuring fairness and transparency. This project is a testament to my expertise in designing and deploying intelligent, decentralized solutions at the forefront of Web3 innovation.



📦 Project Structure




onlyfounders-agents/
├── agent_pitch_strength/
│   ├── nlp_processor.py            # Core NLP logic for pitch analysis
│   ├── pitch_strength_agent.py     # LangGraph orchestration for Pitch Strength Agent
│   └── PitchRegistry.sol           # Solidity smart contract for on-chain pitch scores
├── web/
│   ├── index.html                  # Frontend HTML for pitch upload UI
│   └── script.js                   # Frontend JavaScript for UI interaction and agent simulation
├── .env                            # Environment variables (e.g., API keys)
└── README.md                       # This file




🎯 Current Focus: Pitch Strength Agent (Track 2)



My initial development is concentrated on the Pitch Strength Agent, a cornerstone of this platform. This agent, a prime example of my expertise in building sophisticated AI agents, is meticulously designed to score a founder’s pitch based on narrative clarity, originality, team strength, and market fit. It achieves this through encrypted NLP pipelines, providing an objective assessment of pitch quality while strictly protecting sensitive content. This is accomplished by integrating cutting-edge privacy-preserving technologies like Trusted Execution Environments (TEEs) and Zero-Knowledge Proofs (ZKPs), showcasing my commitment to secure and verifiable AI solutions.



🛠️ Technologies Used



Python: For AI agent development and orchestration.

LangChain / LangGraph: Multi-agent orchestration.

Google Generative AI (Gemini): Core LLM for pitch analysis.

Hugging Face Transformers: For contextual embeddings (originality checks).

pypdf, python-docx: For pitch document parsing.

Solidity: For on-chain smart contracts.

Hardhat / Foundry (Development): For compiling, testing, and deploying contracts.

JavaScript / HTML: For the web-based user interface.

Tailwind CSS: For styling.

Conceptual Privacy Technologies:

Trusted Execution Environments (TEEs): For secure, confidential computation on sensitive data.

Zero-Knowledge Proofs (ZKPs): For verifiable computation without revealing underlying data.

(Future consideration) Homomorphic Encryption (HE): For computation on encrypted data.




🚀 Getting Started




Follow these steps to set up and run the demo locally.

Prerequisites
Python 3.9+

Node.js & npm (or Yarn) for Solidity development tools

Git

1. Clone the Repository
git clone https://github.com/SriAshrai/Pitch_Strength_Onlyfounders.git
cd Pitch_Strength_Onlyfounders


2. Set Up Python Environment
Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt # (You'll create this file next)





Create requirements.txt:




# In the root of the project:
pip freeze > requirements.txt
# Or manually create with:
# python-dotenv
# pypdf
# python-docx
# sentence-transformers
# scikit-learn
# numpy
# google-generativeai
# langchain-google-generativeai
# langchain-core
# langgraph




3. Configure API Keys


 Create a .env file in the onlyfounders-agents/ root directory and add your Google Gemini API key:

# .env file
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"


Note: For running in a Canvas environment, the GEMINI_API_KEY might be automatically provided, but for local development, it's essential.




4. Set Up Solidity Development Environment (Optional for initial demo)



If you plan to compile/deploy the smart contract:

# Choose Hardhat or Foundry
# Using Hardhat:
npm init -y
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox



# Or using Foundry:
curl -L https://foundry.paradigm.xyz | bash
foundryup




5. Running the Demo


The current demo integrates the Python agent logic via a simulation in the JavaScript frontend. For a full interaction, you would typically run a backend service (e.g., Flask/FastAPI) that exposes the Python agent via an API.

For this conceptual demo:

Ensure Python dependencies are installed as per step 2.

Navigate to the web directory:

cd web


Open index.html in your web browser. You can usually do this by dragging the file into your browser, or by using a simple local web server (e.g., Python's http.server):

python -m http.server 8000


Then open http://localhost:8000/index.html in your browser.

Interact with the UI: Upload a .txt, .pdf, or .docx pitch file, or paste pitch text into the textarea. Click "Analyze Pitch" to see simulated results from the Pitch Strength Agent.

Important Note on Simulation: The script.js directly simulates the Python agent's response for demonstration purposes within a browser-only environment. In a production setup, the analyzeBtn would make an actual fetch call to a backend API that wraps and executes the pitch_strength_agent.py.





🤝 Contributing




We welcome contributions to this ambitious project! Please feel free to open issues, submit pull requests, or discuss ideas.




📜 License
