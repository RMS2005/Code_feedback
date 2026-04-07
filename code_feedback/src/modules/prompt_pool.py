# src/modules/prompt_pool.py


"""
This module contains a pool of specialized prompts for the FeedbackEngine.
Each prompt is designed to focus the LLM's analysis on a specific programming
concept (e.g., recursion, loops, error handling). The FeedbackEngine will
select the most relevant prompt based on semantic similarity to the student's
code and/or error messages.
"""

PROMPT_POOL = [
    # --- General & Logic ---
    {
        "id": "general_logic",
        "category": "general",
        "text": "Advice: Provide a general analysis of the provided code's logic and structure in relation to the problem description."
    },
    {
        "id": "logic_conditionals",
        "category": "logic",
        "text": "Advice: Focus on the conditional logic (if/else statements). Check if the conditions accurately capture the problem's requirements."
    },
    {
        "id": "logic_boolean",
        "category": "logic",
        "text": "Advice: Analyze the boolean operators (and, or, not). The logical flow might have a flaw in how conditions are combined."
    },
    {
        "id": "logic_edge_cases",
        "category": "logic",
        "text": "Advice: The logic may not be considering edge cases. Focus your hint on what might happen with empty inputs, zero values, or single-element lists."
    },

    # --- Functions & Scope ---
    {
        "id": "function_design",
        "category": "functions",
        "text": "Advice: Pay attention to the function's design. Evaluate its parameters, return value, and whether it correctly implements its intended purpose."
    },
    {
        "id": "function_return_value",
        "category": "functions",
        "text": "Advice: The function may not be returning a value correctly. Check if all code paths that should return a value actually do."
    },
    {
        "id": "variable_scope",
        "category": "functions",
        "text": "Advice: Examine the scope of variables. A variable might be accessed or modified in a location where it's not defined or holds an unexpected value."
    },

    # --- Loops & Iteration ---
    {
        "id": "loops_general",
        "category": "loops",
        "text": "Advice: Pay attention to the loop constructs. Check for correct initialization, termination conditions, and potential off-by-one or infinite loop errors."
    },
    {
        "id": "loops_off_by_one",
        "category": "loops",
        "text": "Advice: There appears to be an off-by-one error in a loop. Focus the hint on the loop's range or termination condition."
    },
    {
        "id": "loops_infinite",
        "category": "loops",
        "text": "Advice: Suspect an infinite loop. Your hint should guide the student to check the variable that controls the loop's termination."
    },
    {
        "id": "loops_modification_during_iteration",
        "category": "loops",
        "text": "Advice: Analyze if a data structure is being modified while it is being iterated over. This can lead to unpredictable behavior."
    },

    # --- Data Structures ---
    {
        "id": "ds_list_manipulation",
        "category": "data_structures",
        "text": "Advice: Focus on how the list is being manipulated. Check for correct use of methods like append, insert, or remove."
    },
    {
        "id": "ds_dict_access",
        "category": "data_structures",
        "text": "Advice: The logic for accessing or assigning dictionary values seems flawed. Hint towards checking if a key exists before trying to use it."
    },
    {
        "id": "ds_string_concatenation",
        "category": "data_structures",
        "text": "Advice: Analyze the string building logic. The issue may lie in how strings are being concatenated or formatted inside a loop."
    },

    # --- Algorithms ---
    {
        "id": "recursion_general",
        "category": "recursion",
        "text": "Advice: Pay attention to the use of recursion. Analyze the base case, the recursive step, and the likelihood of termination."
    },
    {
        "id": "recursion_base_case",
        "category": "recursion",
        "text": "Advice: The recursive function's base case might be missing or incorrect. Your hint should focus on the condition that stops the recursion."
    },
    {
        "id": "dp_memoization",
        "category": "dynamic_programming",
        "text": "Advice: Pay attention to dynamic programming concepts. Look for evidence of memoization or tabulation to avoid re-computing results."
    },
    {
        "id": "algo_sorting",
        "category": "algorithms",
        "text": "Advice: Focus on the sorting logic. Is the correct sorting key or comparison logic being used?"
    },
    {
        "id": "algo_searching",
        "category": "algorithms",
        "text": "Advice: Examine the search algorithm. The method for finding an element in the data structure may be inefficient or incorrect."
    },

    # --- Error Handling (Specific) ---
    {
        "id": "error_index",
        "category": "error_handling",
        "text": "Advice: The student's error is related to indexing. Focus your hint on how to correctly access elements in a collection, possibly due to an off-by-one error."
    },
    {
        "id": "error_type",
        "category": "error_handling",
        "text": "Advice: The student's error is a type mismatch. Focus your hint on verifying the data types of variables involved in the operation (e.g., adding a string to an integer)."
    },
    {
        "id": "error_zero_division",
        "category": "error_handling",
        "text": "Advice: The student's error is a division by zero. Focus your hint on checking the divisor's value before a division operation."
    },
    {
        "id": "error_key",
        "category": "error_handling",
        "text": "Advice: The code has a KeyError. Your hint should focus on how to safely access dictionary keys, for example, by checking for their existence first."
    },
    {
        "id": "error_name",
        "category": "error_handling",
        "text": "Advice: A NameError occurred. Guide the student to check for misspelled variable or function names, or using a variable before it has been assigned."
    },
    {
        "id": "error_attribute",
        "category": "error_handling",
        "text": "Advice: An AttributeError was raised. The hint should focus on making sure the correct method or attribute is being called on the correct object type."
    },

    # --- OOP & Code Style ---
    {
        "id": "oop_class_design",
        "category": "oop",
        "text": "Advice: Pay attention to the object-oriented design. Analyze the class structure, its attributes, and the constructor's role in initialization."
    },
    {
        "id": "oop_method_usage",
        "category": "oop",
        "text": "Advice: The issue might be in how a class method is used. Check if 'self' is used correctly and if the method is being called on an instance of the class."
    }
]
