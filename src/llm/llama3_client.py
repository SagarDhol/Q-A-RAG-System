import json
from typing import Dict, Any, Optional
import ollama

class Llama3Client:
    def __init__(self, model_name: str = "llama3"):
        """Initialize the Llama3 client."""
        self.model_name = model_name
        
    def generate(self, prompt: str, stream: bool = False, **kwargs):
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to the model
            
        Yields:
            str: The generated text chunks if streaming
            
        Returns:
            str: The complete generated text if not streaming
        """
        if stream:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                **kwargs
            )
            for chunk in response:
                if 'response' in chunk:
                    yield chunk['response']
        else:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                **kwargs
            )
            return response['response']
    
    def generate_structured(self, prompt: str, response_format: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output based on the response format."""
        # Convert the format to a string description
        format_desc = json.dumps(response_format, indent=2)
        
        # Create a system message that enforces the output format
        system_prompt = f"""You are a helpful assistant that always responds with valid JSON.
The response must match the following JSON schema exactly:

{format_desc}

Return only the JSON object, without any additional text or markdown formatting."""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
            
            # Extract the JSON from the response
            response_text = response['message']['content'].strip()
            
            # Sometimes the model might add markdown code blocks
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]  # Remove ```json and ```
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]  # Remove ``` and ```
                
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            # Fallback to text generation if JSON parsing fails
            print(f"Failed to parse JSON response: {e}")
            response = self.generate(prompt)
            return {"error": "Failed to generate structured response", "raw_response": response}
