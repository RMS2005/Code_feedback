import ast

class StaticAnalyzer:
    def _count_nodes(self, tree, node_type):
        """Helper to count specific AST node types."""
        return sum(1 for node in ast.walk(tree) if isinstance(node, node_type))

    def _find_function_defs(self, tree):
        """Finds names of defined functions."""
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    def _check_recursion(self, tree, function_name):
        """Rudimentary check if a function calls itself."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == function_name:
                    # This is a simple check. More robust would need scope analysis.
                    # Check if this call is within the function definition itself.
                    current_func_def = None
                    parent = getattr(node, 'parent', None) # Requires parent pointers (see astor or other libs)
                                                          # For simplicity, we'll skip true recursion check here
                                                          # and focus on other constructs.
                    # A true recursion check is complex with just `ast`.
                    # For now, we'll just note if a function with the same name is called.
                    return True # Placeholder
        return False


    def analyze(self, submission: dict) -> dict:
        student_id = submission['student_id']
        code = submission['code']
        config = submission['config']
        language = submission.get('language', 'python')

        results = {
            "syntax_valid": True,
            "language": language,
            "errors": [],
            "constructs_found": [],
            "metrics": {},
            "style_issues": [],
            "conceptual_warnings": []
        }

        if language != 'python':
            print(f"[STATIC] Skipping AST analysis for non-Python language: {language}")
            results['constructs_found'].append(f"Static analysis for {language} is currently limited to basic validation.")
            if not code.strip():
                results['syntax_valid'] = False
                results['errors'].append("Code is empty.")
            submission['analysis']['static'] = results
            return submission
        
        try:
            tree = ast.parse(code)
            results['metrics']['for_loops'] = self._count_nodes(tree, ast.For)
            results['metrics']['while_loops'] = self._count_nodes(tree, ast.While)
            results['metrics']['if_statements'] = self._count_nodes(tree, ast.If)
            results['metrics']['function_definitions'] = self._count_nodes(tree, ast.FunctionDef)
            
            defined_funcs = self._find_function_defs(tree)
            results['constructs_found'].append(f"Defined functions: {', '.join(defined_funcs) if defined_funcs else 'None'}")

            # Example: Check if expected entry point exists (for function mode)
            exec_mode = config.get('execution_mode', {})
            if exec_mode.get('type') == 'function':
                entry_point = exec_mode.get('entry_point')
                if entry_point and entry_point not in defined_funcs:
                    results['errors'].append(f"Expected function '{entry_point}' not defined.")
                    results['conceptual_warnings'].append(
                        f"The assignment expected you to define a function named '{entry_point}', but it was not found."
                    )
            
            # Simple check for direct `input()` calls (often discouraged if stdin is managed)
            if self._count_nodes(tree, ast.Call) > 0:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'input':
                        results['constructs_found'].append("Direct 'input()' call found.")
                        if exec_mode.get('type') == 'program': # Or if runner handles input
                             results['style_issues'].append(
                                 "Consider if direct 'input()' calls are needed; "
                                 "input is usually provided via standard input (sys.stdin)."
                             )
                        break
            
            # Placeholder for future checks based on concept_inventory
            # e.g., if "recursion" is in inventory, try to detect it.

        except SyntaxError as e:
            results['syntax_valid'] = False
            error_msg = f"Syntax Error: {e.msg} at line {e.lineno}, offset {e.offset}"
            results['errors'].append(error_msg)
            results['constructs_found'] = ["Syntax Error prevents further analysis"]
            print(f"  [STATIC] {student_id}: {error_msg}")
        except Exception as e:
            results['syntax_valid'] = False
            error_msg = f"Unexpected Error during static analysis: {str(e)}"
            results['errors'].append(error_msg)
            results['constructs_found'] = ["Internal Error prevents further analysis"]
            print(f"  [STATIC] {student_id}: {error_msg}")

        submission['analysis']['static'] = results
        print(f"[STATIC] Analysis for {student_id}: For loops: {results['metrics'].get('for_loops', 0)}, Functions: {results['metrics'].get('function_definitions', 0)}")
        return submission