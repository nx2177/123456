import json
import requests

class BaseAgent:
    def __init__(self, model="llama3.2:latest"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
    
    def query_llm(self, prompt, system_prompt="", format_json=False):
        """
        Query the Ollama LLM API with the given prompt
        
        Args:
            prompt (str): The prompt to send to the LLM
            system_prompt (str): Optional system prompt
            format_json (bool): Whether to request the output in JSON format
            
        Returns:
            str: The response from the LLM
        """
        headers = {
            "Content-Type": "application/json"
        }
        
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
    
    def process(self, *args, **kwargs):
        """
        Process method to be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the 'process' method") 