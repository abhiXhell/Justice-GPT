import os
from typing import List, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

class ModelTrainer:
    def __init__(self, model_name: str = "deepseek-r1:latest"):
        self.model_name = model_name
        self.base_model = None
        self.tokenizer = None
        self.trained_model_path = "data/trained_model"
        
    def prepare_training_data(self, pdf_text: str) -> Dataset:
        """Prepare training data from PDF text."""
        # Split text into chunks
        chunks = [chunk.strip() for chunk in pdf_text.split('\n\n') if chunk.strip()]
        
        # Create training examples
        training_data = []
        for chunk in chunks:
            # Create prompt-completion pairs
            prompt = "Question: What does the Indian Labour Act say about "
            completion = f"Answer: {chunk}"
            
            training_data.append({
                "prompt": prompt,
                "completion": completion
            })
        
        # Convert to dataset
        return Dataset.from_list(training_data)
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize the training examples."""
        prompts = examples["prompt"]
        completions = examples["completion"]
        
        # Combine prompt and completion
        texts = [f"{p} {c}" for p, c in zip(prompts, completions)]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return tokenized
    
    def train(self, pdf_text: str):
        """Fine-tune the model on the provided PDF text."""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare model for training
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Get PEFT model
        model = get_peft_model(self.base_model, lora_config)
        
        # Prepare dataset
        print("Preparing training data...")
        dataset = self.prepare_training_data(pdf_text)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.trained_model_path,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_ratio=0.1
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        print("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.trained_model_path)
        
        print("Training completed!")
        
    def load_trained_model(self):
        """Load the trained model and tokenizer."""
        if not os.path.exists(self.trained_model_path):
            raise FileNotFoundError("No trained model found. Please train the model first.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.trained_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.trained_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        return self.base_model, self.tokenizer 