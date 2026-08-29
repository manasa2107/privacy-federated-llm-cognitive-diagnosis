"""
IRT (Item Response Theory) and DINA (Deterministic Input, Noisy-AND gate) Models
For baseline comparison in cognitive diagnosis
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class IRTModel:
    """
    Item Response Theory (IRT) Model
    Uses 2-Parameter Logistic (2PL) model
    
    P(correct|theta, a, b) = 1 / (1 + exp(-a * (theta - b)))
    
    where:
        theta: student ability
        a: item discrimination
        b: item difficulty
    """
    def __init__(self, n_students, n_questions):
        self.n_students = n_students
        self.n_questions = n_questions
        self.theta = None  # Student abilities
        self.a = None      # Item discriminations
        self.b = None      # Item difficulties
    
    def fit(self, student_ids, question_ids, responses, max_iter=20):
        """
        Fit IRT model using Expectation-Maximization
        
        Args:
            student_ids: Array of student IDs
            question_ids: Array of question IDs
            responses: Array of response correctness (0 or 1)
            max_iter: Maximum iterations
        """
        # Initialize parameters
        self.theta = np.random.randn(self.n_students)
        self.a = np.ones(self.n_questions)
        self.b = np.zeros(self.n_questions)
        
        # Create response matrix
        response_matrix = np.full((self.n_students, self.n_questions), np.nan)
        for s, q, r in zip(student_ids, question_ids, responses):
            response_matrix[s, q] = r
        
        # EM algorithm
        for iteration in range(max_iter):
            # E-step: Update student abilities
            for s in range(self.n_students):
                mask = ~np.isnan(response_matrix[s])
                if not mask.any():
                    continue
                
                def neg_log_likelihood(theta_s):
                    p = 1 / (1 + np.exp(-self.a[mask] * (theta_s - self.b[mask])))
                    ll = response_matrix[s, mask] * np.log(p + 1e-10) + \
                         (1 - response_matrix[s, mask]) * np.log(1 - p + 1e-10)
                    return -ll.sum()
                
                result = minimize(neg_log_likelihood, self.theta[s], method='BFGS')
                self.theta[s] = result.x[0]
            
            # M-step: Update item parameters
            for q in range(self.n_questions):
                mask = ~np.isnan(response_matrix[:, q])
                if not mask.any():
                    continue
                
                def neg_log_likelihood(params):
                    a_q, b_q = params
                    p = 1 / (1 + np.exp(-a_q * (self.theta[mask] - b_q)))
                    ll = response_matrix[mask, q] * np.log(p + 1e-10) + \
                         (1 - response_matrix[mask, q]) * np.log(1 - p + 1e-10)
                    return -ll.sum()
                
                result = minimize(neg_log_likelihood, [self.a[q], self.b[q]], 
                                method='BFGS', bounds=[(0.1, 5), (-3, 3)])
                self.a[q], self.b[q] = result.x
            
            if (iteration + 1) % 10 == 0:
                print(f"IRT Iteration {iteration + 1}/{max_iter}")
        
        print("IRT model fitting complete")
    
    def predict_knowledge_states(self, student_ids, n_concepts=4):
        """
        Predict knowledge states from IRT abilities
        
        Since IRT estimates a single ability dimension, we map it to multiple concepts
        by adding noise to create concept-specific estimates
        
        Args:
            student_ids: List of student IDs
            n_concepts: Number of knowledge concepts
            
        Returns:
            knowledge_states: (n_students, n_concepts) array
        """
        if self.theta is None:
            raise ValueError("Model must be fitted first")
        
        # Map theta to [0, 1] using sigmoid
        abilities = 1 / (1 + np.exp(-self.theta[student_ids]))
        
        # Create concept-specific estimates
        # Add small random perturbations to simulate concept variation
        knowledge_states = np.zeros((len(student_ids), n_concepts))
        for i, ability in enumerate(abilities):
            # Base all concepts on ability with small variations
            knowledge_states[i] = np.clip(
                ability + np.random.randn(n_concepts) * 0.1,
                0, 1
            )
        
        return knowledge_states


class DINAModel:
    """
    DINA (Deterministic Input, Noisy-AND gate) Model
    
    Based on De La Torre (2009)
    
    P(correct|eta) = (1 - s)^eta * g^(1-eta)
    
    where:
        eta: Ideal response (AND gate over required concepts)
        s: Slip parameter (P(incorrect | mastered all concepts))
        g: Guess parameter (P(correct | not mastered all concepts))
    """
    def __init__(self, n_students, n_questions, n_concepts, q_matrix):
        """
        Args:
            n_students: Number of students
            n_questions: Number of questions
            n_concepts: Number of knowledge concepts
            q_matrix: (n_questions, n_concepts) binary matrix indicating
                     which concepts are required for each question
        """
        self.n_students = n_students
        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.q_matrix = q_matrix
        
        self.alpha = None  # Student concept mastery (n_students, n_concepts)
        self.s = None      # Slip parameters (n_questions,)
        self.g = None      # Guess parameters (n_questions,)
    
    def fit(self, student_ids, question_ids, responses, max_iter=20):
        """
        Fit DINA model using EM algorithm
        
        Args:
            student_ids: Array of student IDs
            question_ids: Array of question IDs
            responses: Array of response correctness
            max_iter: Maximum iterations
        """
        # Initialize parameters
        self.alpha = (np.random.rand(self.n_students, self.n_concepts) > 0.5).astype(float)
        self.s = np.random.uniform(0.1, 0.3, self.n_questions)
        self.g = np.random.uniform(0.1, 0.3, self.n_questions)
        
        # Create response matrix
        response_matrix = np.full((self.n_students, self.n_questions), np.nan)
        for s, q, r in zip(student_ids, question_ids, responses):
            response_matrix[s, q] = r
        
        for iteration in range(max_iter):
            # E-step: Compute ideal responses
            eta = np.zeros((self.n_students, self.n_questions))
            for s in range(self.n_students):
                for q in range(self.n_questions):
                    # AND gate: student must master ALL required concepts
                    required_concepts = self.q_matrix[q] == 1
                    eta[s, q] = np.all(self.alpha[s, required_concepts] == 1)
            
            # M-step: Update slip and guess parameters
            for q in range(self.n_questions):
                mask = ~np.isnan(response_matrix[:, q])
                if not mask.any():
                    continue
                
                # Students who mastered all concepts (eta = 1)
                mastered = eta[mask, q] == 1
                if mastered.any():
                    # Slip: P(incorrect | mastered)
                    self.s[q] = 1 - response_matrix[mask, q][mastered].mean()
                    self.s[q] = np.clip(self.s[q], 0.01, 0.5)
                
                # Students who didn't master all concepts (eta = 0)
                not_mastered = eta[mask, q] == 0
                if not_mastered.any():
                    # Guess: P(correct | not mastered)
                    self.g[q] = response_matrix[mask, q][not_mastered].mean()
                    self.g[q] = np.clip(self.g[q], 0.01, 0.5)
            
            # Update alpha using maximum likelihood
            for s in range(self.n_students):
                for k in range(self.n_concepts):
                    # Questions requiring concept k
                    relevant_qs = np.where(self.q_matrix[:, k] == 1)[0]
                    mask = ~np.isnan(response_matrix[s, relevant_qs])
                    
                    if not mask.any():
                        continue
                    
                    # Try both alpha=0 and alpha=1, pick better likelihood
                    alpha_orig = self.alpha[s, k]
                    
                    # Try alpha = 1
                    self.alpha[s, k] = 1
                    eta_1 = np.array([np.all(self.alpha[s, self.q_matrix[q] == 1] == 1) 
                                     for q in relevant_qs[mask]])
                    p_1 = np.where(eta_1, 1 - self.s[relevant_qs[mask]], 
                                  self.g[relevant_qs[mask]])
                    ll_1 = (response_matrix[s, relevant_qs[mask]] * np.log(p_1 + 1e-10) + 
                           (1 - response_matrix[s, relevant_qs[mask]]) * np.log(1 - p_1 + 1e-10)).sum()
                    
                    # Try alpha = 0
                    self.alpha[s, k] = 0
                    eta_0 = np.array([np.all(self.alpha[s, self.q_matrix[q] == 1] == 1) 
                                     for q in relevant_qs[mask]])
                    p_0 = np.where(eta_0, 1 - self.s[relevant_qs[mask]], 
                                  self.g[relevant_qs[mask]])
                    ll_0 = (response_matrix[s, relevant_qs[mask]] * np.log(p_0 + 1e-10) + 
                           (1 - response_matrix[s, relevant_qs[mask]]) * np.log(1 - p_0 + 1e-10)).sum()
                    
                    # Pick better one
                    self.alpha[s, k] = 1 if ll_1 > ll_0 else 0
            
            if (iteration + 1) % 10 == 0:
                print(f"DINA Iteration {iteration + 1}/{max_iter}")
        
        print("DINA model fitting complete")
    
    def predict_knowledge_states(self, student_ids):
        """
        Predict knowledge states (concept mastery probabilities)
        
        Args:
            student_ids: List of student IDs
            
        Returns:
            knowledge_states: (n_students, n_concepts) array
        """
        if self.alpha is None:
            raise ValueError("Model must be fitted first")
        
        # DINA gives binary mastery, we return as probabilities
        # Add small noise to avoid exact 0/1 values for MAE comparison
        knowledge_states = self.alpha[student_ids].copy()
        knowledge_states = np.clip(
            knowledge_states + np.random.randn(*knowledge_states.shape) * 0.05,
            0, 1
        )
        
        return knowledge_states


if __name__ == "__main__":
    print("Testing IRT and DINA implementations...")
    
    # Test IRT
    print("\n=== Testing IRT ===")
    n_students = 50
    n_questions = 30
    n_samples = 500
    
    student_ids = np.random.randint(0, n_students, n_samples)
    question_ids = np.random.randint(0, n_questions, n_samples)
    responses = np.random.binomial(1, 0.6, n_samples)
    
    irt_model = IRTModel(n_students, n_questions)
    irt_model.fit(student_ids, question_ids, responses, max_iter=20)
    
    test_students = list(range(10))
    irt_knowledge = irt_model.predict_knowledge_states(test_students, n_concepts=4)
    print(f"IRT Knowledge states shape: {irt_knowledge.shape}")
    print(f"Sample IRT knowledge states:\n{irt_knowledge[:3]}")
    
    # Test DINA
    print("\n=== Testing DINA ===")
    n_concepts = 4
    q_matrix = np.random.binomial(1, 0.5, (n_questions, n_concepts))
    
    dina_model = DINAModel(n_students, n_questions, n_concepts, q_matrix)
    dina_model.fit(student_ids, question_ids, responses, max_iter=20)
    
    dina_knowledge = dina_model.predict_knowledge_states(test_students)
    print(f"DINA Knowledge states shape: {dina_knowledge.shape}")
    print(f"Sample DINA knowledge states:\n{dina_knowledge[:3]}")
    
    print("\nIRT and DINA implementation tests complete!")
