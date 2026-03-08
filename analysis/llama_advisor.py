"""
Llama 3.1 Integration via Groq API
For benchmarking against custom Buffett Transformer model
"""

import os
from groq import Groq

# BUFFETT SYSTEM PROMPT

BUFFETT_SYSTEM_PROMPT = """You are Warren Buffett, the legendary investor and CEO of Berkshire Hathaway. 
Respond to all questions as Warren Buffett would, using his investment philosophy and communication style.

Key principles to embody:
- Value investing: Buy wonderful businesses at fair prices
- Circle of competence: Only invest in what you understand
- Margin of safety: Always leave room for error
- Long-term thinking: "Our favorite holding period is forever"
- Economic moats: Look for durable competitive advantages
- Quality management: Honest and capable leaders matter
- Avoid debt: Great businesses don't need much leverage
- Be fearful when others are greedy, greedy when others are fearful

Communication style:
- Use folksy wisdom and humor
- Reference Omaha and Berkshire Hathaway
- Use simple analogies to explain complex concepts
- Be humble but confident
- Occasionally mention Charlie Munger

Keep responses concise (2-4 sentences) unless asked for detailed explanation."""



# LLAMA ADVISOR CLASS
class LlamaAdvisor:
    """
    Warren Buffett AI using Llama 3.1 via Groq
    """
    
    def __init__(self):
        self.client = None
        self.model = "llama-3.1-8b-instant"
        self.is_loaded = False
        
    def load_model(self):
        """Initialize Groq client — key comes from dashboard.py config"""
        try:
            from config import GROQ_API_KEY
            
            if not GROQ_API_KEY:
                return False, "API key is empty. Update GROQ_API_KEY in config.py"
            
            self.client = Groq(api_key=GROQ_API_KEY)
            self.is_loaded = True
            return True, f"Llama 3.1 loaded via Groq (Model: {self.model})"
        except ImportError:
            return False, "config.py not found. Create config.py with GROQ_API_KEY = 'your-key'"
        except Exception as e:
            return False, f"Error loading Llama: {str(e)}"
    
    def ask(self, question, max_tokens=200):
        """Ask Warren Buffett (Llama) a question"""
        if not self.is_loaded:
            return "Model not loaded. Please load the model first."
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": BUFFETT_SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_stock_opinion(self, ticker, buffett_score, ratios_summary=None):
        """Get Buffett's opinion on a specific stock"""
        if not self.is_loaded:
            return "Model not loaded."
        
        context = f"Stock: {ticker}\nBuffett Score: {buffett_score}/100\n"
        if ratios_summary:
            context += "Key Metrics:\n"
            for key, value in ratios_summary.items():
                context += f"- {key}: {value}\n"
        
        prompt = f"""Based on this analysis of {ticker}:

{context}

As Warren Buffett, give your opinion on this stock in 2-3 sentences. 
Consider whether it meets your criteria for a wonderful business at a fair price."""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": BUFFETT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=150,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def compare_response(self, question, custom_response):
        """Generate a response and return both for comparison"""
        llama_response = self.ask(question)
        return {
            "question": question,
            "custom_model": custom_response,
            "llama_model": llama_response
        }



# BENCHMARKING CLASS
class ModelBenchmark:
    """Benchmark custom model against Llama 3.1"""
    
    def __init__(self, custom_chatbot, llama_advisor):
        self.custom = custom_chatbot
        self.llama = llama_advisor
        self.results = []
    
    def run_benchmark(self, questions=None):
        if questions is None:
            questions = [
                "Who are you?",
                "What is value investing?",
                "How do you pick stocks?",
                "What makes a good business?",
                "What is your advice for young investors?",
                "How do you think about risk?",
                "What is a margin of safety?",
                "What is an economic moat?",
                "Should I invest in index funds?",
                "How do you value a company?",
            ]
        
        self.results = []
        for q in questions:
            custom_response = self.custom.ask(q) if self.custom.is_loaded else "Model not loaded"
            llama_response = self.llama.ask(q) if self.llama.is_loaded else "Model not loaded"
            self.results.append({
                "question": q,
                "custom_model": custom_response,
                "llama_model": llama_response
            })
        return self.results
    
    def get_summary(self):
        if not self.results:
            return {"error": "No benchmark results. Run benchmark first."}
        
        custom_avg_len = sum(len(r["custom_model"]) for r in self.results) / len(self.results)
        llama_avg_len = sum(len(r["llama_model"]) for r in self.results) / len(self.results)
        
        return {
            "total_questions": len(self.results),
            "custom_model": {
                "name": "Handcoded Transformer (18M params)",
                "avg_response_length": round(custom_avg_len, 1),
            },
            "llama_model": {
                "name": "Llama 3.1 8B via Groq",
                "avg_response_length": round(llama_avg_len, 1),
            }
        }


# GETTER : creates fresh instance each time (no stale caching)
def get_llama_advisor():
    """Get a new Llama advisor instance"""
    return LlamaAdvisor()