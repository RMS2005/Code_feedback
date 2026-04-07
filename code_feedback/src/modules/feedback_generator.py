from pathlib import Path
import csv
import os

class FeedbackGenerator:

    def generate_individual_report_string(self, submission: dict) -> tuple:
        """
        Generates the feedback report string for a single student AND returns
        the structured data needed for other reports (like CSV).
        """
        student_id = submission['student_id']
        full_code = submission.get('code', 'No code available.')
        
        # --- Dynamic Analysis Data ---
        dynamic_results = submission['analysis'].get('dynamic', [])
        total_tests = len(dynamic_results)
        passed_count = sum(1 for r in dynamic_results if r.get('status') == 'pass')

        feedback_results = submission['analysis'].get('feedback', {})
        technical_summary = feedback_results.get('technical_summary')
        concept_score = feedback_results.get('concept_score', 0)
        concept_reason = feedback_results.get('concept_reason')
        auto_fix = feedback_results.get('auto_fix')
        analyzed_construct = feedback_results.get('summarized_construct', 'N/A')
        
        evaluation_results = submission['analysis'].get('evaluation', {})
        similarity_score = evaluation_results.get('semantic_similarity_score', 0)
        
        # --- Build the Markdown Report String ---
        report_lines = []
        report_lines.append(f"\n\n--- Feedback Report for: {student_id} ---")
        report_lines.append(f"Assignment: {submission['config']['assignment_id']}")
        
        # Static Analysis Section
        report_lines.append("\n--- STATIC ANALYSIS (Code Structure & Potential Issues) ---\n")
        static_results = submission['analysis'].get('static', {})
        if not static_results.get('syntax_valid', False):
            report_lines.append("Your code has one or more Syntax Errors.")
            for err in static_results.get('errors', []): report_lines.append(f"- {err}")
        else:
            report_lines.append("Your code has valid syntax.")
            metrics = static_results.get('metrics', {})
            report_lines.append(f"- For Loops: {metrics.get('for_loops', 0)}")
            report_lines.append(f"- Function Definitions: {metrics.get('function_definitions', 0)}")
            # ... Add other static metrics if you want ...

            report_lines.append(f"\n**Standard Tests: {passed_count} / {total_tests} passed.**")

        # Edge Case Results
        edge_results = [r for r in dynamic_results if r.get('is_edge')]
        if edge_results:
            report_lines.append("\n--- EDGE CASE ANALYSIS ---\n")
            edge_passed = sum(1 for r in edge_results if r.get('status') == 'pass')
            for result in edge_results:
                status = result.get('status', 'ERROR').upper()
                report_lines.append(f"- Edge '{result.get('name', 'N/A')}': {status}")
                if status == 'RUNTIME_ERROR':
                    error_log = result.get('error', '')
                    if error_log: report_lines.append(f"  - Runtime Error Details: \n```\n{error_log}\n```")
                elif status == 'COMPILATION_ERROR':
                    comp_log = result.get('error', '')
                    if comp_log: report_lines.append(f"  - Compilation Error Details: \n```\n{comp_log}\n```")
            report_lines.append(f"\n**Edge Cases: {edge_passed} / {len(edge_results)} passed.**")

        # Feedback Engine (LLM) Section
        # AI Evaluation Section
        report_lines.append("\n--- ADVANCED AI EVALUATION ---\n")
        report_lines.append(f"- **Semantic Similarity (BERT Score)**: {similarity_score:.4f}")
        report_lines.append(f"- **Concept Understanding Score**: {concept_score}/10")
        if concept_reason:
            report_lines.append(f"  - Reasoning: {concept_reason}")

        if technical_summary:
            report_lines.append(f"\nTechnical Summary (for '{analyzed_construct}'):")
            report_lines.append(f"```\n{technical_summary}\n```")

        if auto_fix:
            report_lines.append(f"\n**AI Suggested Fix:**")
            report_lines.append(f"```\n{auto_fix}\n```")

        # Plagiarism Section
        plagiarism_alerts = submission['analysis'].get('plagiarism', [])
        if plagiarism_alerts:
            report_lines.append("\n\n> [!CAUTION]")
            report_lines.append("> **POTENTIAL PLAGIARISM DETECTED**")
            for alert in plagiarism_alerts:
                report_lines.append(f"> - High semantic similarity ({alert['score']:.4f}) with student: **{alert['partner']}**")

        # Return all necessary data points as a tuple
        return (
            "\n".join(report_lines), 
            student_id, 
            passed_count, 
            total_tests, 
            technical_summary,
            full_code,
            similarity_score,
            concept_score
        )

    def generate_aggregated_report(self, all_submissions: list, output_dir_str: str, assignment_id: str):
        """Generates a single aggregated Markdown report for all students."""
        print(f"✔️ [FEEDBACK] Generating aggregated markdown report...")
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_filename = f"Report_{assignment_id.replace(' ', '_')}.md"
        output_filepath = output_dir / report_filename

        full_report_content = [f"# Autograder - Aggregated Feedback Report", f"## Assignment: {assignment_id}\n"]
        if not all_submissions:
            full_report_content.append("No submissions were processed.")
        else:
            for submission in all_submissions:
                # Unpack the tuple; we only need the report string for this method
                individual_report_str, *args = self.generate_individual_report_string(submission)
                full_report_content.append(individual_report_str)
                full_report_content.append("\n" + "="*80 + "\n")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(full_report_content))
        print(f"   => Aggregated markdown report saved to {output_filepath}")

    def generate_csv_summary(self, all_submissions: list, output_dir_str: str, assignment_id: str):
        """Generates a single CSV summary of all submissions."""
        print(f"✔️ [FEEDBACK] Generating CSV summary...")
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = f"Summary_{assignment_id.replace(' ', '_')}.csv"
        output_filepath = output_dir / csv_filename

        headers = ['student_id', 'tests_passed', 'tests_total', 'score_percentage', 'bert_similarity', 'concept_score', 'semantic_summary']
        processed_data_for_csv = []
        
        if not all_submissions:
            print("   => No data to write to CSV.")
            return

        for submission in all_submissions:
            try:
                # Get all data points from the helper function
                individual_tuple = self.generate_individual_report_string(submission)
                _, student_id, passed_count, total_tests, summary, code_snippet, similarity, concept = individual_tuple
                
                score_percentage = (passed_count / total_tests * 100) if total_tests > 0 else 0
                
                summary_for_csv = summary.replace('\n', ' ') if summary else "N/A"
                
                processed_data_for_csv.append({
                    'student_id': student_id,
                    'tests_passed': passed_count,
                    'tests_total': total_tests,
                    'score_percentage': f"{score_percentage:.2f}",
                    'bert_similarity': f"{similarity:.4f}",
                    'concept_score': concept,
                    'semantic_summary': summary_for_csv
                })
            except Exception as e:
                # Catch any unexpected error during data processing for a single student
                student_id = submission.get('student_id', 'UnknownStudent')
                print(f"   [WARNING] Could not process data for CSV for student {student_id}. Error: {e}")
                # Optionally add an error row to the CSV
                processed_data_for_csv.append({
                    'student_id': student_id,
                    'tests_passed': 'N/A', 'tests_total': 'N/A', 'score_percentage': 'N/A',
                    'semantic_summary': "ERROR",
                    'code_snippet': f"Error processing submission data: {e}"
                })

        # Write all collected data to the CSV file
        try:
            with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(processed_data_for_csv)
            print(f"   => CSV summary saved to {output_filepath}")
        except Exception as e_write:
            print(f"   [ERROR] Failed to write CSV file at {output_filepath}. Error: {e_write}")

    def generate_all_reports(self, all_submissions: list, output_dir_str: str, assignment_id: str):
        """Generates all available report types (Markdown and CSV)."""
        if not all_submissions:
            print("[FEEDBACK] No submissions provided, skipping report generation.")
            return
            
        self.generate_aggregated_report(all_submissions, output_dir_str, assignment_id)
        self.generate_csv_summary(all_submissions, output_dir_str, assignment_id)