from typing import List, Dict
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from .model_trainer import ModelTrainer

class LLMInterface:
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.trained_model_path = "data/trained_model"
        
        # Check if model is already trained
        if os.path.exists(self.trained_model_path):
            try:
                self.model, self.tokenizer = self.model_trainer.load_trained_model()
                self.use_trained_model = True
            except Exception as e:
                print(f"Error loading trained model: {e}")
                self.use_trained_model = False
        else:
            self.use_trained_model = False
        
        # Fallback to Ollama if no trained model
        if not self.use_trained_model:
            self.llm = Ollama(model="deepseek-r1:latest")
        
        self.system_prompt = """
You are a helpful, supportive, and knowledgeable AI assistant specializing in Indian Labour and Consumer Court law. Your job is to help real people with their legal questions in a friendly, conversational, and clear way—never as a textbook.

When answering:
- Use a warm, human tone. For example, start with phrases like "Don't worry," or "Yes, you absolutely can take legal action," etc.
- Clearly explain the legal basis, but in simple language.
- Always suggest specific next steps (e.g., "You can contact your local Labour Commissioner," or "File a complaint online at https://labour.gov.in or https://consumerhelpline.gov.in").
- Remind the user to keep documentation (e.g., chat records, payment receipts, messages) as proof.
- If relevant, suggest free legal aid via the District Legal Services Authority (DLSA) or the National Consumer Helpline (1800-11-4000).
- Structure your answer in three sections: Incident, Legal Basis, Suggested Actions.
- Be concise, supportive, and practical.

Context: {context}

Question: {question}

Answer (in the above style):
"""
        
        self.prompt = PromptTemplate(
            template=self.system_prompt,
            input_variables=["context", "question"]
        )
        
        if not self.use_trained_model:
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt
            )
    
    def train_model(self, pdf_text: str) -> None:
        """Train the model on the provided PDF text."""
        try:
            self.model_trainer.train(pdf_text)
            self.model, self.tokenizer = self.model_trainer.load_trained_model()
            self.use_trained_model = True
        except Exception as e:
            print(f"Error during training: {e}")
            self.use_trained_model = False
    
    def generate_response(self, question: str, context: str) -> str:
        """Generate a response using either the trained model or Ollama."""
        try:
            if self.use_trained_model:
                # Use the fine-tuned model
                inputs = self.tokenizer(
                    f"{self.system_prompt.format(context=context, question=question)}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response.strip()
            else:
                # Fallback to Ollama
                return self.chain.run(
                    question=question,
                    context=context
                ).strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def is_relevant_question(self, question: str) -> bool:
        """Check if the question is relevant to Indian Labour and Consumer Court law."""
        # List of keywords for Indian Labour and Consumer Court law
        domain_keywords = [
            "labour", "labor", "wages", "employment", "employee", "employer", "industrial dispute",
            "trade union", "gratuity", "bonus", "minimum wage", "workman", "layoff", "retrenchment",
            "consumer", "consumer court", "consumer forum", "consumer protection", "defective product",
            "service deficiency", "refund", "compensation", "unfair trade practice", "consumer rights",
            "installation", "amazon", "e-commerce", "purchase", "warranty", "guarantee"
        ]
        try:
            # LLM-based check
            relevance_check = """Analyze if the following question is related to Indian Labour and Consumer Court law. Respond with only 'yes' or 'no'.\n\nQuestion: {question}"""
            prompt = PromptTemplate(
                template=relevance_check,
                input_variables=["question"]
            )
            if self.use_trained_model:
                inputs = self.tokenizer(
                    relevance_check.format(question=question),
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                chain = LLMChain(llm=self.llm, prompt=prompt)
                response = chain.run(question=question)
            response = response.strip().lower()
            if "yes" in response:
                return True
            if "no" in response:
                # Fallback to keyword check
                for kw in domain_keywords:
                    if kw in question.lower():
                        return True
                return False
            # If ambiguous, fallback to keyword check
            for kw in domain_keywords:
                if kw in question.lower():
                    return True
            return False
        except Exception as e:
            print(f"Error checking relevance: {e}")
            # Fallback to keyword check
            for kw in domain_keywords:
                if kw in question.lower():
                    return True
            return True  # Default to True if there's an error 