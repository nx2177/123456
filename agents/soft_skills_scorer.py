import json
import re
import requests

class OllamaLLM:
    """
    LLM handler for Ollama API
    """
    def __init__(self, model="llama3.2:latest"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        
    def query(self, prompt, system_prompt="", format_json=False):
        """
        Query the Ollama LLM API
        
        Args:
            prompt (str): The prompt to send to the LLM
            system_prompt (str): Optional system prompt
            format_json (bool): Whether to request the output in JSON format
            
        Returns:
            str: The response from the LLM
        """
        headers = {"Content-Type": "application/json"}
        
        if format_json:
            if system_prompt:
                system_prompt += " Your response MUST be valid JSON format only, with no additional text, no code block markers, and no explanation."
            else:
                system_prompt = "Your response MUST be valid JSON format only, with no additional text, no code block markers, and no explanation."
            
            # Add JSON format instructions to the prompt as well
            prompt = f"{prompt}\n\nIMPORTANT: Respond with ONLY the JSON object. No introductions, explanations, or code block markers."
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_text = response.json()["response"]
            
            # Clean the response if JSON is expected
            if format_json:
                # Remove code block markers if present
                response_text = response_text.replace("```json", "").replace("```", "").strip()
                
                # Remove any text before the first { and after the last }
                first_brace = response_text.find('{')
                last_brace = response_text.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    response_text = response_text[first_brace:last_brace+1]
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying LLM: {e}")
            return ""


class SoftSkillsScorer:
    """
    Agent D: Score the candidate's soft skills
    """
    def __init__(self, llm_model="llama3.2:latest"):
        """
        Initialize the soft skills scorer agent
        
        Args:
            llm_model (str): The LLM model to use
        """
        self.llm = OllamaLLM(model=llm_model)
    
    def _extract_json_from_text(self, text):
        """
        Extract JSON from text that might contain extra content
        
        Args:
            text (str): Text that might contain JSON
            
        Returns:
            dict: Extracted JSON as dict or empty dict if not found
        """
        # Try to find JSON-like content enclosed in braces
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        json_matches = re.findall(json_pattern, text)
        
        if json_matches:
            # Try each match until we find valid JSON
            for potential_json in json_matches:
                try:
                    return json.loads(potential_json)
                except json.JSONDecodeError:
                    continue
        
        return {}
    
    def process(self, resume_text, job_description):
        """
        Evaluate soft skills evidence in the resume compared to job requirements
        
        Args:
            resume_text (str): The full resume text
            job_description (str): The job description text
            
        Returns:
            tuple: (score, justification) where score is between 0 and 1
        """
        system_prompt = """
        You are a specialized HR analyst who evaluates soft skills in resumes.
        Soft skills include communication, teamwork, leadership, problem-solving, adaptability,
        time management, conflict resolution, creativity, emotional intelligence, etc.
        
        Your task is to:
        1. Identify soft skills required in the job description
        2. Find evidence of these skills in the resume
        3. Evaluate how well the candidate's soft skills match the job requirements
        4. Provide a score between 0.0 (no match) and 1.0 (perfect match)
        5. Justify your score with specific examples from the resume
        """
        
        prompt = f"""
        Please analyze the following resume and job description to evaluate the candidate's soft skills.
        
        RESUME:
        {resume_text}
        
        JOB DESCRIPTION:
        {job_description}
        
        Provide your assessment as a JSON with two fields:
        1. "score": a number between 0.0 and 1.0
        2. "justification": 2-3 sentences explaining your reasoning with specific examples from the resume
        """
        
        try:
            # First attempt with format_json flag
            response = self.llm.query(prompt, system_prompt, format_json=True)
            
            # Try to parse the response as JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                result = self._extract_json_from_text(response)
                
                # If still no valid JSON, make a second attempt with stronger instructions
                if not result:
                    stronger_prompt = f"""
                    Evaluate the candidate's soft skills against the job description requirements.
                    
                    RESUME:
                    {resume_text}
                    
                    JOB DESCRIPTION:
                    {job_description}
                    
                    Return ONLY a valid JSON object with exactly these two fields:
                    {{
                        "score": 0.0 to 1.0,
                        "justification": "Your 2-3 sentence explanation here with examples"
                    }}
                    """
                    response = self.llm.query(stronger_prompt, system_prompt, format_json=True)
                    try:
                        result = json.loads(response)
                    except json.JSONDecodeError:
                        result = self._extract_json_from_text(response)
            
            # Ensure we have valid score and justification
            if not result or "score" not in result:
                return 0.5, "Unable to generate proper justification due to technical issues."
                
            score = float(result.get("score", 0.5))
            # Cap score between 0 and 1
            score = max(0.0, min(score, 1.0))
            justification = result.get("justification", "No justification provided.")
            
            return score, justification
            
        except Exception as e:
            print(f"Error in soft skills scoring: {str(e)}")
            # Return default values if there's an issue
            return 0.5, "Unable to generate proper justification due to technical issues." 