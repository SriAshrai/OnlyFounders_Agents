import os
from dotenv import load_dotenv
from typing import Dict, Any, List
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import asyncio

# Correct and recommended way to import and configure Google Generative AI SDK
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables from .env file
load_dotenv()

class NLPProcessor:
    def __init__(self):
        # Configure Gemini API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("NLPProcessor: GEMINI_API_KEY not found in .env file. Please set it up.")
            genai.configure(api_key="") # Will be filled by Canvas runtime or you provide your key
        else:
             genai.configure(api_key=self.gemini_api_key)

        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("NLPProcessor: Loaded Sentence Transformer model for originality.")

        # --- IMPORTANT CHANGE: Using gemini-1.5-flash for broader availability ---
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=self.gemini_api_key) 
        self.embeddings_llm = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=self.gemini_api_key)

        self.clarity_prompt = PromptTemplate(
            template="""You are an expert pitch evaluator.
            Assess the clarity and structure of the following startup pitch.
            Score it on a scale of 1 to 10 (10 being perfectly clear and well-structured).
            Provide a brief reasoning for the score.
            Format your response as a JSON object with keys 'score' (integer) and 'reasoning' (string).

            Pitch:
            {pitch_text}
            """,
            input_variables=["pitch_text"],
        )

        self.team_strength_prompt = PromptTemplate(
            template="""You are an expert pitch evaluator focused on team assessment.
            Analyze the following pitch for explicit and implicit indicators of team strength (experience, relevant background, cohesion, previous successes).
            Score it on a scale of 1 to 10 (10 being an exceptionally strong team).
            Provide a brief reasoning for the score.
            Format your response as a JSON object with keys 'score' (integer) and 'reasoning' (string).

            Pitch:
            {pitch_text}
            """,
            input_variables=["pitch_text"],
        )

        self.market_fit_prompt = PromptTemplate(
            template="""You are an expert pitch evaluator focused on market fit.
            Evaluate the following pitch for its understanding of the market, problem-solution fit, and competitive landscape.
            Score it on a scale of 1 to 10 (10 being an outstanding market fit).
            Provide a brief reasoning for the score.
            Format your response as a JSON object with keys 'score' (integer) and 'reasoning' (string).

            Pitch:
            {pitch_text}
            """,
            input_variables=["pitch_text"],
        )

        self.json_parser = JsonOutputParser()
        print("NLPProcessor: Initialized LLM and scoring prompts.")


    async def _call_llm_for_score(self, prompt: PromptTemplate, pitch_text: str) -> Dict[str, Any]:
        """Helper to call LLM with a specific prompt and parse JSON output."""
        try:
            chain = prompt | self.llm | self.json_parser
            response = await chain.ainvoke({"pitch_text": pitch_text})
            print(f"NLPProcessor: LLM raw response for prompt {prompt.template[:30]}: {json.dumps(response, indent=2)}")
            return response
        except Exception as e:
            print(f"NLPProcessor: Error calling LLM for scoring: {e}")
            return {"score": 0, "reasoning": f"LLM error: {e}. Check API key and model availability."}

    def _extract_text_from_doc(self, file_path: str) -> str:
        """Extracts text from PDF, DOCX, or TXT files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        text = ""
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file format. Only PDF, DOCX, and TXT are supported.")
        return text

    async def analyze_pitch_content(self, pitch_content: str, llm_for_analysis: ChatGoogleGenerativeAI, embeddings_model_for_analysis: GoogleGenerativeAIEmbeddings) -> Dict[str, Any]:
        """
        Analyzes a pitch for clarity, team strength, market fit, and originality.
        This function is designed to be called by the agent with the pre-loaded pitch text.
        """
        if not pitch_content:
            return {"error": "No pitch content provided for analysis."}

        processed_pitch_text = pitch_content[:10000] # Truncate for context window and memory

        scores = {
            "overall_score": None, # Will be calculated below
            "components": {}
        }

        print("NLPProcessor: Calling LLM for clarity score...")
        clarity_result = await self._call_llm_for_score(self.clarity_prompt, processed_pitch_text)
        scores['components']['clarity'] = clarity_result

        print("NLPProcessor: Calling LLM for team strength score...")
        team_strength_result = await self._call_llm_for_score(self.team_strength_prompt, processed_pitch_text)
        scores['components']['team_strength'] = team_strength_result

        print("NLPProcessor: Calling LLM for market fit score...")
        market_fit_result = await self._call_llm_for_score(self.market_fit_prompt, processed_pitch_text)
        scores['components']['market_fit'] = market_fit_result

        print("NLPProcessor: Calculating originality score...")
        try:
            existing_pitches_corpus = [
                "Our decentralized finance protocol revolutionizes lending and borrowing on blockchain.",
                "AI-powered medical diagnostics for early disease detection using patient data.",
                "A platform connecting artists and fans in the metaverse through NFTs.",
                "Revolutionizing retail with AR/VR shopping experiences.",
                "Building a new social network focused on privacy and user ownership of data.",
                "A sustainable energy solution leveraging advanced solar panel technology.",
            ]
            pitch_embedding = embeddings_model_for_analysis.embed_query(processed_pitch_text)
            corpus_embeddings = await embeddings_model_for_analysis.aembed_documents(existing_pitches_corpus)

            pitch_embedding_np = np.array(pitch_embedding).reshape(1, -1)
            corpus_embeddings_np = np.array(corpus_embeddings)

            similarities = cosine_similarity(pitch_embedding_np, corpus_embeddings_np)
            max_similarity = np.max(similarities)
            
            originality_score = max(1, int(10 - (max_similarity * 9)))
            scores['components']['originality'] = {
                'score': originality_score,
                'reasoning': f"Pitch similarity to existing concepts: {max_similarity:.2f}. Lower similarity indicates higher originality."
            }
        except Exception as e:
            print(f"NLPProcessor: Error calculating originality score: {e}")
            scores['components']['originality'] = {'score': 0, 'reasoning': f"Error: {e}"}


        # --- Calculate Overall Score (within NLPProcessor) ---
        total_score_sum = 0
        num_scored_components = 0
        for comp in ['clarity', 'team_strength', 'market_fit', 'originality']:
            # Safely access the score, defaulting to 0 if not found or None
            component_score = scores['components'].get(comp, {}).get('score', 0)
            if component_score is not None: # Ensure it's not explicitly None
                total_score_sum += component_score
                num_scored_components += 1

        if num_scored_components > 0:
            scores['overall_score'] = round(total_score_sum / num_scored_components)
        else:
            scores['overall_score'] = 0 # Default to 0 if no valid scores

        print(f"NLPProcessor: Final analysis scores for the pitch: {json.dumps(scores, indent=2)}")
        return scores

# Example Usage (for testing the NLPProcessor directly) - This block is not used by the agents.
async def test_nlp_processor():
    processor = NLPProcessor()

    dummy_pitch_content_good = """
    Our startup, Quantum Leap Solutions, aims to revolutionize the supply chain industry using a novel blend of AI and quantum computing.
    The team consists of Dr. Alice Smith, a leading expert in quantum algorithms, and Mr. Bob Johnson, a seasoned supply chain veteran with 15 years of experience at global logistics firms.
    We identified a critical bottleneck in real-time inventory management, causing billions in losses annually. Our solution provides predictive analytics with unparalleled accuracy, reducing waste by 30%.
    The market for supply chain optimization is growing rapidly, with a projected value of $50 billion by 2030. We have secured early partnerships with two major manufacturing companies.
    """
    scores_good = await processor.analyze_pitch_content(
        pitch_content=dummy_pitch_content_good,
        llm_for_analysis=processor.llm,
        embeddings_model_for_analysis=processor.embeddings_llm
    )
    print(json.dumps(scores_good, indent=2))

if __name__ == "__main__":
    asyncio.run(test_nlp_processor())

