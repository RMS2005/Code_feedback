import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .embedding_engine import EmbeddingEngine
try:
    from bert_score import score as bert_score_func
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

class EvaluationEngine:
    def __init__(self, embedding_engine: EmbeddingEngine = None):
        self.embedding_engine = embedding_engine or EmbeddingEngine()

    def calculate_semantic_similarity(self, student_code: str, reference_code: str) -> float:
        """
        Calculates the semantic similarity between student code and reference code
        using the fine-tuned embedding model.
        """
        if not student_code or not reference_code:
            return 0.0

        try:
            student_emb = self.embedding_engine.get_code_embedding(student_code)
            ref_emb = self.embedding_engine.get_code_embedding(reference_code)

            if student_emb and ref_emb:
                student_vec = np.array(student_emb).reshape(1, -1)
                ref_vec = np.array(ref_emb).reshape(1, -1)
                similarity = cosine_similarity(student_vec, ref_vec)[0][0]
                return float(similarity)
        except Exception as e:
            print(f"[EVALUATION] Error calculating cosine semantic similarity: {e}")
        
        return 0.0

    def calculate_bert_score(self, student_code: str, reference_code: str) -> float:
        """
        Uses the bert-score library to calculate F1 score between student and reference code.
        """
        if not BERT_SCORE_AVAILABLE or not student_code or not reference_code:
            return 0.0
        
        try:
            # We use 'en' as language for code-related tokens in many BERT models
            # Alternatively, we could use a code-specific model here if needed.
            P, R, F1 = bert_score_func([student_code], [reference_code], lang="en", verbose=False)
            return float(F1.mean())
        except Exception as e:
            print(f"  [EVALUATION] BERTScore library failed: {e}")
            return 0.0

    def analyze(self, submission: dict) -> dict:
        """
        Enriches the submission with semantic evaluation metrics.
        """
        try:
            student_id = submission['student_id']
            config = submission['config']
            student_code = submission.get('code', '')
            reference_code = config.get('reference_solution', '')

            print(f"[EVALUATION] Running semantic evaluation for {student_id}...")

            if 'evaluation' not in submission['analysis']:
                submission['analysis']['evaluation'] = {}

            # 1. Cosine Similarity (Embedding-based)
            cosine_sim = self.calculate_semantic_similarity(student_code, reference_code)
            
            # 2. BERTScore (Token-matching-based semantic similarity)
            bert_sim = self.calculate_bert_score(student_code, reference_code)
            
            # We'll use a weighted blend or just BERTScore as the primary metric
            # User explicitly asked for BERT score to be the core feature.
            final_similarity_score = bert_sim if bert_sim > 0 else cosine_sim

            submission['analysis']['evaluation'].update({
                'semantic_similarity_score': round(final_similarity_score, 4),
                'cosine_similarity': round(cosine_sim, 4),
                'bert_score': round(bert_sim, 4),
                'has_reference': bool(reference_code)
            })

            if not reference_code:
                print(f"  [EVALUATION] WARNING: No reference solution provided in config for similarity check.")

            print(f"  [EVALUATION] Semantic Similarity Score: {final_similarity_score:.4f}")
        except Exception as e:
            print(f"  [EVALUATION] CRITICAL ERROR during analysis: {e}")
            # Ensure evaluation dictionary exists at least
            if 'evaluation' not in submission['analysis']:
                submission['analysis']['evaluation'] = {}
            submission['analysis']['evaluation']['semantic_similarity_score'] = 0.0
            
        return submission
