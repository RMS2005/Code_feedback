import numpy as np
from sklearn.manifold import TSNE
import pandas as pd # For organizing data for Plotly
from pathlib import Path

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[ANALYTICS_ENGINE] WARNING: umap-learn library not found. t-SNE will be used.")

# Import Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[ANALYTICS_ENGINE] WARNING: plotly library not found. Cannot generate interactive plots.")


class AnalyticsEngine:
    def __init__(self):
        print("[ANALYTICS_ENGINE] Initializing Analytics Engine...")

    def _prepare_data_for_plot(self, all_submissions: list) -> pd.DataFrame | None:
        """
        Extracts embeddings and test results, returning a structured Pandas DataFrame.
        """
        plot_data = []

        for sub in all_submissions:
            embedding = sub.get('analysis', {}).get('embedding', {}).get('code_embedding')
            dynamic_results = sub.get('analysis', {}).get('dynamic', [])
            
            if not (embedding and isinstance(embedding, list)):
                print(f"  [ANALYTICS_ENGINE] Skipping submission for {sub.get('student_id', 'Unknown')} due to missing/invalid embedding.")
                continue

            # Calculate pass percentage
            pass_percentage = 0.0
            if dynamic_results:
                total_tests = len(dynamic_results)
                if total_tests > 0:
                    passed_count = sum(1 for r in dynamic_results if r.get('status') == 'pass')
                    pass_percentage = (passed_count / total_tests) * 100
            
            # New: Semantic and Concept evaluation
            evaluation = sub.get('analysis', {}).get('evaluation', {})
            feedback = sub.get('analysis', {}).get('feedback', {})
            
            similarity_score = evaluation.get('semantic_similarity_score', 0.0)
            concept_score = feedback.get('concept_score', 0)

            # Prepare code snippet for hover text (first N lines)
            full_code = sub.get('code', 'No code available.')
            # Use &lt; and &gt; for HTML safety in hover
            code_snippet_for_hover = "<br>".join(full_code.replace("<", "&lt;").replace(">", "&gt;").splitlines()[:15])
            if len(full_code.splitlines()) > 15:
                code_snippet_for_hover += "<br>... (truncated)"

            plot_data.append({
                "student_id": sub.get('student_id', 'Unknown'),
                "pass_percentage": pass_percentage,
                "bert_similarity": similarity_score,
                "concept_score": concept_score,
                "embedding": embedding,
                "code_snippet": code_snippet_for_hover,
                "plagiarism_flag": False # Default
            })

        if not plot_data:
            return None
            
        df = pd.DataFrame(plot_data)
        
        # --- Plagiarism Detection Logic ---
        print("  [ANALYTICS_ENGINE] Checking for pairwise plagiarism...")
        embeddings = np.array(df['embedding'].tolist())
        if len(embeddings) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)
            
            plagiarism_threshold = 0.95
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if sim_matrix[i][j] > plagiarism_threshold:
                        print(f"    [WARNING] High similarity ({sim_matrix[i][j]:.4f}) detected between {df.iloc[i]['student_id']} and {df.iloc[j]['student_id']}")
                        df.at[i, 'plagiarism_flag'] = True
                        df.at[j, 'plagiarism_flag'] = True
        
        return df

    def generate_interactive_embedding_plot(self, plot_df: pd.DataFrame, 
                                            output_path: Path, assignment_id: str, 
                                            method: str = "umap"):
        """
        Generates and saves an interactive 2D scatter plot of code embeddings using Plotly.
        """
        num_samples = len(plot_df)
        if num_samples < 2:
            print("  [ANALYTICS_ENGINE] Not enough valid submissions (need at least 2) to generate a plot.")
            return
            
        # Get the high-dimensional embeddings from the DataFrame
        embeddings_array = np.array(plot_df['embedding'].tolist())

        print(f"  [ANALYTICS_ENGINE] Reducing dimensionality using {method} with {num_samples} samples...")
        low_dim_embeddings = None

        # --- Dimensionality Reduction (UMAP or t-SNE) ---
        # (This logic for choosing and running UMAP/t-SNE is the same as before)
        if method.lower() == "umap" and UMAP_AVAILABLE:
            try:
                # Adjust UMAP parameters for small N
                n_neighbors = max(2, min(15, num_samples - 1))
                min_dist = 0.1 if num_samples > 5 else 0.5
                n_epochs = 200 # Explicitly set for small N for stability
                
                print(f"    [ANALYTICS_ENGINE] UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}, n_epochs={n_epochs}")
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                                    n_epochs=n_epochs, random_state=42)
                low_dim_embeddings = reducer.fit_transform(embeddings_array)
            except Exception as e_umap:
                print(f"    [ANALYTICS_ENGINE] UMAP failed: {e_umap}. Falling back to t-SNE.")
                method = "tsne" # Force t-SNE
                low_dim_embeddings = None

        if low_dim_embeddings is None and method.lower() == "tsne":
            try:
                perplexity = float(num_samples - 1) if num_samples <= 5 else min(30.0, float(num_samples - 1))
                perplexity = max(1.0, perplexity)
                
                print(f"    [ANALYTICS_ENGINE] t-SNE params: perplexity={perplexity}")
                tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', 
                            init='pca', random_state=42)
                low_dim_embeddings = tsne.fit_transform(embeddings_array)
            except Exception as e_tsne:
                print(f"    [ANALYTICS_ENGINE] t-SNE failed: {e_tsne}. Cannot generate plot.")
                return

        if low_dim_embeddings is None:
            print("  [ANALYTICS_ENGINE] Dimensionality reduction failed. Cannot generate plot.")
            return

        # Add the 2D coordinates to our DataFrame
        plot_df['x'] = low_dim_embeddings[:, 0]
        plot_df['y'] = low_dim_embeddings[:, 1]
        
        print(f"  [ANALYTICS_ENGINE] Generating interactive Plotly graph...")

        import plotly.express as px

        # --- Create Interactive Plot with Plotly Express ---

        # Set to False for a light theme
        dark_theme = False

        # 'Inferno', 'Viridis', 'Plasma', 'Cividis' are good perceptually uniform colorscales
        # that work well on both light and dark backgrounds. 'Inferno' goes from black/purple to yellow.
        # 'Cividis' is designed to be friendly for color vision deficiency.

        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='plagiarism_flag', # Changed from pass_percentage to highlight plagiarism
            color_discrete_map={True: 'red', False: 'blue'},
            hover_name='student_id',
            hover_data={
                'x': False,
                'y': False,
                'pass_percentage': ':.2f',
                'bert_similarity': ':.4f',
                'concept_score': True,
                'plagiarism_flag': True,
                'code_snippet': True
            },
            title=f"Code Submission Semantic Clusters (Red indicates Plagiarism Flag) for: {assignment_id}",
            labels={
                'x': f'{method.upper()} Dimension 1',
                'y': f'{method.upper()} Dimension 2',
                'pass_percentage': 'Pass %'
            }
        )

        # Marker styling
        fig.update_traces(
            marker=dict(size=15, line=dict(width=1, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )

        # Layout
        fig.update_layout(
            legend_title_text='Pass Percentage',
            plot_bgcolor='rgba(17,17,17,1)' if dark_theme else 'white',
            paper_bgcolor='rgba(17,17,17,1)' if dark_theme else 'white',
            font_color='white' if dark_theme else 'black',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            coloraxis_colorbar=dict(
                title="Pass %",
                ticks="outside",
                dtick=20
            )
        )
        # Update hover template for better formatting
        fig.update_traces(
            hovertemplate="<b>Student ID:</b> %{hovertext}<br>" +
                          "<b>Pass Percentage:</b> %{marker.color:.2f}%<br>" +
                          "<b>BERT Similarity:</b> %{customdata[0]:.4f}<br>" +
                          "<b>Concept Score:</b> %{customdata[1]}/10<br>" +
                          "<br><b>Code Snippet:</b><br>%{customdata[2]}" +
                          "<extra></extra>", # Hides the "trace" box
            customdata=plot_df[['bert_similarity', 'concept_score', 'code_snippet']] # Pass data to customdata
        )


        plot_filename = f"interactive_embeddings_{assignment_id.replace(' ', '_').replace('/', '_')}_{method}.html"
        plot_filepath = output_path / plot_filename
        try:
            fig.write_html(plot_filepath)
            print(f"  [ANALYTICS_ENGINE] Interactive embedding plot saved to: {plot_filepath}")
        except Exception as e_save:
            print(f"  [ANALYTICS_ENGINE] Error saving interactive plot: {e_save}")
        

    def generate_report(self, all_submissions: list, output_path_str: str, assignment_id: str):
        """
        Main method for generating instructor analytics.
        """
        print(f"[ANALYTICS_ENGINE] Generating analytics for assignment: {assignment_id}")
        
        if not PLOTLY_AVAILABLE:
            print("  [ANALYTICS_ENGINE] Plotly library not found. Skipping plot generation.")
            return

        output_path = Path(output_path_str)
        output_path.mkdir(parents=True, exist_ok=True)

        if not all_submissions:
            print("  [ANALYTICS_ENGINE] No submissions to analyze.")
            return

        # Prepare data in a pandas DataFrame
        plot_dataframe = self._prepare_data_for_plot(all_submissions)

        if plot_dataframe is not None and not plot_dataframe.empty:
            plot_method = "umap" if UMAP_AVAILABLE else "tsne"
            self.generate_interactive_embedding_plot(plot_dataframe, output_path, assignment_id, method=plot_method)
        else:
            print("  [ANALYTICS_ENGINE] No valid data available to generate embedding plot.")
        
        print("[ANALYTICS_ENGINE] Analytics report generation complete.")
