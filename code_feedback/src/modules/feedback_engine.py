import torch
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, AutoModelForCausalLM # Corrected for Qwen2
import ast
import numpy as np
import astor # For robust function source code extraction
from .embedding_engine import EmbeddingEngine
from .prompt_pool import PROMPT_POOL
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
OLLAMA_MODEL_ID = "phi3:mini"

def ollama_generate(model_tag: str, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    """
    Invokes a model served by Ollama.
    """
    print(f"  [OLLAMA_CLIENT] Calling model '{model_tag}' via Ollama...")
    try:
        llm = ChatOllama(
            model=model_tag,
            temperature=temperature,
            repetition_penalty=1.3,
            request_timeout=60,  # Increased timeout for larger models
            thinking=False,
        )
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
        import re
        raw_output = llm.invoke(messages).content
        ai_msg_content = re.sub(r"<thinking>.*?</thinking>", "", raw_output, flags=re.DOTALL)

        print(f"  [OLLAMA_CLIENT] Received response from Ollama.")
        return ai_msg_content
    except Exception as e:
        print(f"[OLLAMA_CLIENT] ERROR: Ollama call failed for model '{model_tag}'. Details: {e}")
        return f"Error: Could not get response from Ollama model {model_tag}."

class FeedbackEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SEMANTIC_ENGINE] Initializing SemanticEngine on device: {self.device}")
        print("[FEEDBACK_ENGINE] Initializing internal engines for prompt selection...")
        self.embedding_engine = EmbeddingEngine()
        self.prompt_pool_data = []
        self._initialize_prompt_pool()
        

    def _initialize_prompt_pool(self):
        """
        Loads prompts from the pool, generates their embeddings, and stores them for later use.
        """
        if not self.embedding_engine or not self.embedding_engine.model:
            print("[FEEDBACK_ENGINE] WARNING: Embedding model not available. Prompt selection will be disabled.")
            return

        print(f"[FEEDBACK_ENGINE] Embedding {len(PROMPT_POOL)} prompts from the prompt pool...")
        for prompt_item in PROMPT_POOL:
            try:
                embedding = self.embedding_engine.get_code_embedding(prompt_item["text"])
                if embedding:
                    self.prompt_pool_data.append({
                        "id": prompt_item["id"],
                        "text": prompt_item["text"],
                        "embedding": np.array(embedding)
                    })
                else:
                    print(f"  [FEEDBACK_ENGINE] WARNING: Could not generate embedding for prompt id: {prompt_item['id']}")
            except Exception as e:
                print(f"  [FEEDBACK_ENGINE] ERROR embedding prompt id {prompt_item['id']}: {e}")
        
        if self.prompt_pool_data:
            print(f"[FEEDBACK_ENGINE] Successfully embedded {len(self.prompt_pool_data)} prompts.")

    def _find_best_prompt(self, target_embedding: list | np.ndarray) -> str:
        """
        Finds the most relevant prompt from the pool based on cosine similarity.
        """
        if not self.prompt_pool_data or target_embedding is None:
            return ""

        try:
            target_vector = np.array(target_embedding).reshape(1, -1)
            prompt_vectors = np.array([item["embedding"] for item in self.prompt_pool_data])
            similarities = cosine_similarity(target_vector, prompt_vectors)
            best_prompt_index = np.argmax(similarities)
            best_prompt = self.prompt_pool_data[best_prompt_index]

            print(f"  [FEEDBACK_ENGINE] Best matching prompt selected: '{best_prompt['id']}' (Similarity: {similarities[0][best_prompt_index]:.4f})")
            return best_prompt["text"]
        except Exception as e:
            print(f"  [FEEDBACK_ENGINE] ERROR during prompt selection: {e}")
            return ""
    def get_technical_summary(self, code_snippet: str, error_message: str, language: str, question: str) -> str | None:
        """
        Generates a technical summary and, if errors are present, debugging feedback.
        """
        # Prompt Engineering is key here. This system prompt is quite detailed.
    
        system_prompt = '''
            You are a highly specialized AI programming debugger. You will be given two input fields:
            - <input_1>: The student's code snippet and any runtime error it produced.
            - <input_2>: An advice related to code debugging.

            Your Task:
            Generate a new <output> that provides a single, concise debugging insight.

            Guidelines:
            2. <input_1> provides the student's attempt and its erroneous outcome.
            3. <input_2> provides an advice you must adhere to.
            4. Your goal is to create an <output> that acts as a targeted hint, by understanding the student's code and reason of failure (`<input_1>`)  by following the advice ('<input_2>').

            Purpose and Context:
            - The output should focus on the most likely conceptual error or logical flaw, not just the surface-level syntax error.
            - Ensure the output helps the student identify their mistake without giving away the direct solution.

            Key Principles:
            - Do NOT simply rephrase the problem or restate the runtime error.
            - Identify the core logical disconnect hinted at by the runtime error in the context of the problem.
            - Infer the likely misconception (e.g., off-by-one error, incorrect data type handling, flawed algorithm).

            How to Construct the Output:
            1. Analyze the code submission (`<input_1>`) to understand the required logic.
            2. Analyze the student's code and runtime error (`<input_2>`) to find the deviation from the required logic.
            3. Formulate a precise, focused, and contextually relevant hint that guides the student to the source of the error.

            Output Requirements:
            - The final output must be a single, concise sentence.
            - The output must be enclosed in <output> and </output> tags.
            - Avoid redundancy or irrelevant details.
            '''
        error_message_text = error_message if error_message else "No runtime errors were observed during execution."
        best_prompt_text = ""
        if self.embedding_engine:
            code_embedding = self.embedding_engine.get_code_embedding(code_snippet)
            if code_embedding:
                best_prompt_text = self._find_best_prompt(code_embedding)
        user_prompt = f"""
        Inputs:

        <input_1>
        Student's Code Snippet:
        ```python
        {code_snippet}
        Runtime Error:
        {error_message_text}
        </input_1>
        
        <input_2>
        Advice:
        {best_prompt_text}
        </input_2>

        Output:
        <output>

        """
        print(f"  [FEEDBACK_ENGINE] Requesting technical summary from Ollama...")
        try:
            summary = ollama_generate(OLLAMA_MODEL_ID, system_prompt, user_prompt)
            if "Error: Could not get response" in summary or not summary.strip():
                raise Exception("Ollama unreachable or empty response")
        except Exception as e:
            print(f"  [FEEDBACK_ENGINE] WARNING: Ollama failed ({e}). Using mock summary.")
            return f"Mock Summary: The code implements a {language} solution for the requested task. Logic appears sound but requires manual verification."
        
        # Simple extraction from <output> tags
        import re
        match = re.search(r"<output>(.*?)</output>", summary, re.DOTALL)
        return match.group(1).strip() if match else summary.strip()

    def get_concept_score(self, code_snippet: str, language: str, assignment_goal: str) -> tuple[int, str]:
        """
        Uses LLM to evaluate the student's conceptual understanding (0-10) and reason.
        """
        system_prompt = f"""
        You are a programming instructor evaluating a student's submission in {language}.
        The assignment goal is: {assignment_goal}

        Evaluate ONLY the student's conceptual understanding (e.g., proper recursion, algorithm logic).
        Ignore minor syntax errors if the logic is correct.

        Output format:
        <score>integer between 0 and 10</score>
        <reason>one-sentence explanation</reason>
        """
        response = "Mock: <score>7</score><reason>Ollama unreachable. Using default concept score.</reason>"
        try:
            response = ollama_generate(OLLAMA_MODEL_ID, system_prompt, user_prompt)
        except Exception:
            pass

        import re
        score_match = re.search(r"<score>(\d+)</score>", response)
        reason_match = re.search(r"<reason>(.*?)</reason>", response)
        
        score = int(score_match.group(1)) if score_match else 5
        reason = reason_match.group(1).strip() if reason_match else "Could not determine score."
        
        return score, reason

    def get_code_fix(self, code_snippet: str, error_message: str, language: str) -> str:
        """
        Uses LLM to provide a small code fix and explanation.
        """
        system_prompt = f"""
        You are a debugger. The student's {language} code failed with: {error_message}
        Provide a 3-4 line corrected snippet and a 1-sentence fix explanation.
        
        Output format:
        <fix_code>corrected snippet</fix_code>
        <fix_reason>one-sentence explanation</fix_reason>
        """
        user_prompt = f"Student Code:\n{code_snippet}"
        
        response = ollama_generate(OLLAMA_MODEL_ID, system_prompt, user_prompt)
        
        import re
        fix_match = re.search(r"<fix_code>(.*?)</fix_code>", response, re.DOTALL)
        reason_match = re.search(r"<fix_reason>(.*?)</fix_reason>", response)
        
        fix_code = fix_match.group(1).strip() if fix_match else "No fix generated."
        fix_reason = reason_match.group(1).strip() if reason_match else "N/A"
        
        return f"{fix_reason}\n\nSuggested Fix:\n{fix_code}"
    
    def _extract_function_code(self, full_code: str, function_name: str) -> str | None:
        """Extracts the source code of a specific function using AST parsing and astor."""
        try:
            tree = ast.parse(full_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    function_source = astor.to_source(node) # Convert AST node back to source
                    return function_source.strip()
            print(f"    [SEMANTIC_ENGINE_DEBUG] Function '{function_name}' not found in AST for extraction.")
            return None
        except SyntaxError as se: # Catch syntax errors that prevent parsing
            print(f"    [SEMANTIC_ENGINE_DEBUG] SyntaxError parsing code during extraction of '{function_name}': {se}")
            return None
        except Exception as e: # Catch other errors like issues with astor
            print(f"    [SEMANTIC_ENGINE_DEBUG] Error during AST-based extraction of '{function_name}': {e}")
            return None
            
    def _get_defined_function_names(self, full_code: str) -> list[str]:
        """Parses code and returns a list of top-level function definition names."""
        names = []
        try:
            tree = ast.parse(full_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    names.append(node.name)
        except SyntaxError: # If code has syntax error, can't get functions
            pass 
        except Exception: # Other parsing issues
            pass
        return list(set(names)) # Use set to get unique names

    def analyze(self, submission: dict) -> dict:
        """
        Performs semantic analysis: code embedding, technical summarization,
        and rudimentary error feedback using Ollama for generative tasks.
        """
        student_id = submission['student_id']
        config = submission['config']
        full_code = submission.get('code', '')
        language = config.get('language', 'python') # Default to python if not specified
        question = config.get('question', None)

        print(f"[SEMANTIC_ENGINE] Analyzing semantics for {student_id}...")
        if 'feedback' not in submission['analysis']:
            submission['analysis']['feedback'] = {}
        
        # Initialize/update semantic analysis results structure
        submission['analysis']['feedback'].update({
            'generative_model': OLLAMA_MODEL_ID, # Model used via Ollama
            'technical_summary': None,
            'summarized_construct': "N/A",
            'error_explanation': None, 
            'identified_concepts': [],
            'concept_score': 0,
            'concept_reason': None,
            'auto_fix': None
        })
        all_errors = []
        dynamic_results = submission['analysis'].get('dynamic', [])
        for test_result in dynamic_results:
            if test_result.get('status') == 'runtime_error':
                err = test_result.get('error', '')
                if err and err not in all_errors:
                    all_errors.append(err)
        error_message = "\n".join(all_errors)


        # 2. Generate Technical Summary using Ollama
        if full_code:
            code_to_summarize = None
            summarized_construct_name = "N/A"
            exec_mode_config = config.get('execution_mode', {})
            defined_function_names = self._get_defined_function_names(full_code)

            # Logic to select the most relevant code snippet for summarization
            if exec_mode_config.get("type") == "function":
                entry_point = exec_mode_config.get("entry_point")
                if entry_point:
                    print(f"  [SEMANTIC_ENGINE] Attempting to extract function '{entry_point}' for summarization...")
                    extracted_code = self._extract_function_code(full_code, entry_point)
                    if extracted_code: 
                        code_to_summarize, summarized_construct_name = extracted_code, f"function: {entry_point}"
                    else: print(f"    [SEMANTIC_ENGINE] Could not extract '{entry_point}' for summarization.")
            
            if not code_to_summarize: # If not function mode or extraction failed
                if defined_function_names:
                    non_main_functions = [f_name for f_name in defined_function_names if f_name.lower() != "main"]
                    if non_main_functions:
                        target_func_name = non_main_functions[0] # Prioritize first non-main
                        print(f"  [SEMANTIC_ENGINE] Prioritizing non-main function '{target_func_name}' for summarization...")
                        extracted_code = self._extract_function_code(full_code, target_func_name)
                        if extracted_code: code_to_summarize, summarized_construct_name = extracted_code, f"function: {target_func_name}"
                    elif "main" in defined_function_names or "Main" in defined_function_names: # Fallback to main
                        main_func_name = "main" if "main" in defined_function_names else "Main"
                        print(f"  [SEMANTIC_ENGINE] No distinct non-main functions, attempting to summarize '{main_func_name}'...")
                        extracted_code = self._extract_function_code(full_code, main_func_name)
                        if extracted_code: code_to_summarize, summarized_construct_name = extracted_code, f"function: {main_func_name}"
            
            if not code_to_summarize: # Final fallback to full code
                code_to_summarize, summarized_construct_name = full_code, "full_code (fallback)"
            
            if code_to_summarize:
                print(f"    [SEMANTIC_ENGINE_DEBUG] Code snippet for summarization ({summarized_construct_name}):\n{code_to_summarize[:300]}...\n------")
                summary = self.get_technical_summary(code_to_summarize, error_message,language, question) 
                if summary:
                    submission['analysis']['feedback']['technical_summary'] = summary
                    submission['analysis']['feedback']['summarized_construct'] = summarized_construct_name
                    print(f"  [FEEDBACK_ENGINE] Concept Scoring for {student_id}...")
                    
                    # New: Concept Scoring
                    goal = config.get("problem_statement", "Implement the requested algorithm correctly.")
                    c_score, c_reason = self.get_concept_score(code_to_summarize, language, goal)
                    submission['analysis']['feedback']['concept_score'] = c_score
                    submission['analysis']['feedback']['concept_reason'] = c_reason
                    
                    # New: Auto-Fix if error exists
                    if error_message:
                        submission['analysis']['feedback']['auto_fix'] = self.get_code_fix(code_to_summarize, error_message, language)

                    print(f"  [FEEDBACK_ENGINE] Concept Score: {c_score}/10, Reason: {c_reason}")
                else:
                    print(f"  [FEEDBACK_ENGINE] Failed to generate feedback via Ollama.")
            else: # This case should be rare if full_code is the ultimate fallback
                print(f"  [SEMANTIC_ENGINE] No code identified/available to summarize for {student_id}.")
        elif not full_code:
            print(f"  [SEMANTIC_ENGINE] No code provided by submission to summarize for {student_id}.")
            
            
        return submission

