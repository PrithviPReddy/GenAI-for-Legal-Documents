from typing import List, Dict
import re
import json
import traceback
import google.generativeai as genai
from app.utils.logger import logger
import time 
class ImprovedLLMProcessor:
    """Enhanced LLM processor with better prompting and context handling"""

    # In app/services/llm_processor.py
    def analyze_text_for_risks(self, text: str) -> list:
        """
        Analyzes text for a predefined checklist of financial and legal risks
        using a single API call.
        """
        logger.info("Starting single-call risk analysis...")

        # A new, comprehensive prompt for a single API call
        single_call_prompt = f"""
You are an expert legal document analyst. Your task is to analyze the provided document text and identify any clauses that fall into the specific risk categories listed below.

**Instructions:**
1.  Carefully read the document text.
2.  For each risk category, determine if a relevant clause exists in the text.
3.  If you find one or more risks, construct a JSON object with a single key "risks". This key should contain a list of JSON objects, where each object represents a single identified risk.
4.  Each risk object must have the following three keys: "risk_category", "explanation", and "quote".
5.  If NO risks from the categories are found, return a JSON object with an empty list: {{"risks": []}}.

**Risk Categories to Scan For:**
-   **Automatic Renewal:** Clauses mentioning auto-renewal or automatic continuation of services/contracts.
-   **High Penalties or Unclear Fees:** Mentions of specific penalties, late fees, termination fees, or vague language like 'administrative fees'.
-   **Waiver of Rights / Arbitration:** Language where a party waives their right to sue, join a class action, or demands mandatory arbitration.
-   **One-Sided Indemnification:** "Hold harmless" clauses that unfairly favor one party.
-   **Exclusions & Limitations of Liability:** Clauses that limit the provider's responsibility (e.g., 'we are not responsible for...').
-   **Unfavorable Payment Terms:** Terms like variable interest rates, prepayment penalties, or acceleration clauses in loans.
-   **Ambiguous or Vague Language:** Clauses with subjective or poorly defined terms like 'at our sole discretion' or 'subject to change without notice'.
-   **Restrictions on Use or Access:** Significant restrictions on how a property, service, or product can be used.
-   **Data Privacy & Sharing:** Clauses about collecting, using, or sharing personal data with third parties.

**Example of a valid JSON response:**
{{
  "risks": [
    {{
      "risk_category": "Automatic Renewal",
      "explanation": "The contract will automatically renew for another year if not cancelled 30 days prior.",
      "quote": "This agreement shall automatically renew for successive one-year terms unless terminated in writing by either party at least thirty (30) days prior to the end of the then-current term."
    }}
  ]
}}

--- DOCUMENT TEXT ---
{text[:25000]}
"""

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(single_call_prompt)

            # Extract JSON from the response, supporting markdown code blocks
            json_str_match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
            if json_str_match:
                json_str = json_str_match.group(1)
            else:
                # Fallback for plain JSON
                json_start = response.text.find("{")
                json_end = response.text.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    json_str = response.text[json_start:json_end]
                else:
                    raise ValueError("No JSON object found in the response.")

            parsed_response = json.loads(json_str)
            
            # Validate the structure of the parsed response
            if isinstance(parsed_response, dict) and "risks" in parsed_response and isinstance(parsed_response["risks"], list):
                found_risks = parsed_response["risks"]
                logger.info(f"Risk analysis complete. Found {len(found_risks)} potential risks.")
                return found_risks
            else:
                logger.warning("LLM response had invalid structure. Expected a dict with a 'risks' list.")
                return [] # Return empty list on structure mismatch

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from LLM response for risk analysis: {e}")
            logger.debug(f"Raw response was: {response.text}")
            return [] # Return empty list on parsing failure (Option A)
        except Exception as e:
            logger.error(f"An unexpected error occurred during risk analysis: {e}")
            return []
   
    

    def summarize_text(self, text: str, chunker=None) -> str:
        """
        Summarizes a large text using a single API call, designed for models
        with large context windows.
        """
        logger.info("Starting single-call summarization process...")

        # Stage 1: Simple safety limit. We will build the advanced fallback later.
        # A 300,000 character limit is a safe proxy for a ~100k token window.
        if len(text) > 300000:
            logger.warning(f"Document with {len(text)} characters exceeds the simple summarizer limit.")
            # This is where we will later add the advanced, token-aware fallback logic.
            return "Error: This document is too large for the standard summarizer. Advanced summarization is not yet implemented."

        prompt = f"""
You are an expert legal analyst. Your task is to provide a clear and effective summary of the following legal document.

**Instructions:**
1.  Begin with a concise introductory paragraph that explains the document's overall purpose and identifies the key parties involved.
2.  Follow the paragraph with a bulleted list highlighting the most critical points, obligations, and clauses for the user to be aware of.
3.  Ensure the language is simple and easy for a non-lawyer to understand.

**Document Text to Summarize:**
---
{text}
---
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            final_summary = response.text.strip()
            
            logger.info("Successfully generated summary in a single call.")
            return final_summary

        except Exception as e:
            logger.error(f"Failed to summarize text in a single call: {e}")
            logger.error(traceback.format_exc())
            return f"Error during summarization: {str(e)}"

    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        # Use a modern, capable model
        self.model_name = model_name
        self.system_prompt = """
[SYSTEM INSTRUCTION]
You are an expert AI legal assistant specializing in document analysis. Your goal is to provide precise, factual answers based strictly on the text provided.

## Primary Task
Your task is to answer a list of questions based *exclusively* on the document excerpts provided in the [CONTEXT] section. Synthesize information across the text chunks as needed to form a complete answer.

## Critical Rules
1.  **Strictly Grounded:** Your entire answer MUST be derived from the provided [CONTEXT]. Do not use any external knowledge or information from your training data.
2.  **Handling Unanswerable Questions:** If the answer to a question cannot be found in the [CONTEXT], you MUST respond with the exact phrase: "The provided text does not contain enough information to answer this question." Do not apologize or attempt to answer anyway.
3.  **Be Concise:** Provide direct answers. Do not use introductory phrases like "According to the provided text..." or "The document states that...".
4.  **Match Question Count:** You must provide one answer for every question asked. The number of strings in the final "answers" list must be the same as the number of questions.

## Output Format
Your response MUST be a single, valid JSON object. Do not include any explanatory text, apologies, or markdown formatting like ```json before or after the JSON object.

### JSON Structure Example
{
  "answers": [
    "Answer to question 1.",
    "Answer to question 2.",
    "The provided text does not contain enough information to answer this question."
  ]
}
"""
    
    def generate_answers(self, questions: List[str], context_chunks: List[str]) -> List[str]:
        """Generate answers with improved context handling and logging"""
        try:
            context = self.format_context(context_chunks)
            
            # Log payload
            logger.info(f" Sending to LLM:")
            logger.info(f"  - Questions: {len(questions)}")
            logger.info(f"  - Context chunks: {len(context_chunks)}")
            logger.info(f"  - Total context length: {len(context)} chars")
            logger.info(f"  - Model: {self.model_name}")
            
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            user_message = f"""CONTEXT CHUNKS:
{context}

QUESTIONS TO ANSWER:
{questions_text}

Please answer each question based only on the provided context chunks. Look for both direct information and related concepts that can help answer the questions."""
            
            logger.info("Prompt preview (first 500 chars): " + (user_message[:500] + "..." if len(user_message) > 500 else user_message))
            
            # Call Gemini
            logger.info("Making Gemini API call...")
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(f"{self.system_prompt}\n\n{user_message}")
            response_text = response.text.strip()
            
            logger.info(f" Raw LLM Response (first 500 chars): {response_text[:500]}...")
            
            parsed_answers = self.parse_response(response_text, questions)
            
            logger.info(f" Generated answers for {len(questions)} questions:")
            for i, answer in enumerate(parsed_answers, 1):
                logger.info(f"  {i}. {answer}")
            
            return parsed_answers
            
        except Exception as e:
            logger.error(f" Failed to generate answers: {e}")
            logger.error(traceback.format_exc())
            return [f"Error: {str(e)}" for _ in questions]
    
    def format_context(self, chunks: List[str]) -> str:
        """Format context chunks for better LLM understanding"""
        return "\n\n".join([f"[Chunk {i+1}]\n{chunk.strip()}" for i, chunk in enumerate(chunks)])
    
    def parse_response(self, response_text: str, questions: List[str]) -> List[str]:
        """Parse LLM response with improved error handling"""
        try:
            # Extract JSON (supporting code-block wrapping)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_start = response_text.find("{")
                json_str = response_text[json_start:] if json_start != -1 else None
            
            if not json_str:
                raise ValueError("No JSON object found in response")
            
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict) or "answers" not in parsed:
                raise ValueError("Invalid JSON structure: 'answers' missing")
            
            answers = parsed["answers"]
            if not isinstance(answers, list):
                raise ValueError("'answers' must be a list")
            
            # Pad/truncate to match question count
            while len(answers) < len(questions):
                answers.append("Unable to find relevant information in the provided context.")
            return answers[:len(questions)]
        
        except Exception as err:
            logger.warning(f"JSON parsing failed: {err}")
            return self.fallback_parse(response_text, questions)
    
    def fallback_parse(self, response_text: str, questions: List[str]) -> List[str]:
        """Fallback parsing when JSON fails"""
        answers = []
        current_answer = ""
        
        for line in response_text.splitlines():
            line = line.strip()
            if not line or line.startswith(("```", "{", "}", '"answers"', "CONTEXT", "QUESTIONS")):
                continue
            
            if re.match(r"^\d+\.", line):
                if current_answer:
                    answers.append(current_answer.strip())
                current_answer = re.sub(r"^\d+\.\s*", "", line)
            elif current_answer:
                current_answer += " " + line
            else:
                current_answer = line
        
        if current_answer:
            answers.append(current_answer.strip())
        
        while len(answers) < len(questions):
            answers.append("Unable to process this question due to response parsing issues.")
        
        return answers[:len(questions)]

