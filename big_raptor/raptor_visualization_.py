# raptor_visualization.py
import os
import json
import webbrowser
import threading
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler


class RaptorVisualizationHandler:
    """
    Handler class for generating visualizations of the Raptor document hierarchy.
    """

    @staticmethod
    def generate_hierarchy_data(retriever, namespace: str = ""):
        """Generate structured data representing the document hierarchy."""
        hierarchy_data = {
            "levels": {},
            "total_documents": 0,
            "level_distribution": {},
            "metadata": {
                "tree_depth": retriever.tree_depth,
                "clustering_threshold": retriever.clustering_threshold,
                "max_length_in_cluster": retriever.max_length_in_cluster,
                "mode": retriever.mode
            }
        }

        # Get level distribution
        level_distribution = retriever._get_level_distribution(namespace)
        hierarchy_data["level_distribution"] = level_distribution

        # Calculate total documents
        total_docs = sum(count for level, count in level_distribution.items()
                         if isinstance(level, int))
        hierarchy_data["total_documents"] = total_docs

        # Sample documents from each level for visualization
        for level in range(retriever.tree_depth + 1):
            if level not in level_distribution or level_distribution[level] == 0:
                continue

            # Get sample docs from this level
            sample_size = min(10, level_distribution[level])
            try:
                docs = retriever._get_documents_by_metadata("level", level, namespace=namespace)
                sample_docs = docs[:sample_size] if docs else []

                # Extract relevant info
                level_docs = []
                for doc in sample_docs:
                    doc_info = {
                        "id": doc.id or doc.metadata.get("document_id", "unknown"),
                        "content_preview": doc.page_content[:200] + "..." if len(
                            doc.page_content) > 200 else doc.page_content,
                        "metadata": {k: v for k, v in doc.metadata.items()
                                     if k not in ["embedding"] and not isinstance(v, (list, np.ndarray))}
                    }

                    # For level > 0, get child count
                    if level > 0 and "child_count" in doc.metadata:
                        doc_info["child_count"] = doc.metadata["child_count"]

                    level_docs.append(doc_info)

                hierarchy_data["levels"][level] = {
                    "document_count": level_distribution[level],
                    "sample_documents": level_docs
                }
            except Exception as e:
                print(f"Error getting documents for level {level}: {str(e)}")
                hierarchy_data["levels"][level] = {
                    "document_count": level_distribution[level],
                    "sample_documents": [],
                    "error": str(e)
                }

        return hierarchy_data


    @staticmethod
    def create_html_visualization(hierarchy_data, output_path, namespace=""):
        """Create an HTML visualization of the hierarchy using template concatenation."""
        os.makedirs(output_path, exist_ok=True)

        # Use a URL-friendly filename - replace hyphens with underscores
        safe_namespace = namespace.replace("-", "_") if namespace else "default"
        base_name = f"raptor_{safe_namespace}"

        data_filename = os.path.join(output_path, f"{base_name}_data.json")
        html_filename = os.path.join(output_path, f"{base_name}.html")

        print(f"Creating visualization at: {html_filename}")

        # Save hierarchy data as JSON for the visualization
        with open(data_filename, 'w') as f:
            json.dump(hierarchy_data, f, indent=2)

        # Build the HTML content section by section to avoid nested f-strings

        # Start with the basic structure
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Raptor Visualization - {namespace}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .level {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .level-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .document {{ margin-bottom: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 3px; }}
                .stats {{ background-color: #eef; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Raptor Document Hierarchy</h1>

            <div class="stats">
                <h2>Statistics</h2>
                <p>Total Documents: {hierarchy_data["total_documents"]}</p>
                <p>Tree Depth: {hierarchy_data["metadata"]["tree_depth"]}</p>
                <p>Level Distribution:</p>
                <ul>
        """

        # Add level distribution list items
        for level, count in hierarchy_data["level_distribution"].items():
            if isinstance(level, int):
                html_content += f"        <li>Level {level}: {count} documents</li>\n"

        # Continue with the document hierarchy
        html_content += """
                </ul>
            </div>

            <h2>Document Hierarchy</h2>
        """

        # Add each level
        for level, level_data in sorted(hierarchy_data["levels"].items(), key=lambda x: int(x[0]), reverse=True):
            is_root = level == hierarchy_data["metadata"]["tree_depth"]
            root_label = " (Root)" if is_root else ""

            html_content += f"""
            <div class="level">
                <div class="level-title">Level {level}{root_label}</div>
                <p>{level_data["document_count"]} documents</p>
            """

            # Add sample documents
            sample_docs = level_data.get("sample_documents", [])[:5]
            for doc in sample_docs:
                html_content += f"""
                <div class="document">
                    <p><strong>Content:</strong> {doc["content_preview"]}</p>
                    <p><small>ID: {doc["id"]}</small></p>
                </div>
                """

            # Add "more documents" message if needed
            if level_data["document_count"] > 5:
                more_count = level_data["document_count"] - 5
                html_content += f"""
                <p><small>+ {more_count} more documents...</small></p>
                """

            # Close the level div
            html_content += """
            </div>
            """

        # Close the HTML
        html_content += """
        </body>
        </html>
        """

        with open(html_filename, 'w') as f:
            f.write(html_content)

        print(f"Visualization files created:")
        print(f"- HTML: {html_filename}")
        print(f"- Data: {data_filename}")

        return html_filename


class VisualizationServer:
    """Simple HTTP server to serve Raptor visualizations."""

    def __init__(self, directory, port=8080):
        self.directory = os.path.abspath(directory)  # Use absolute path
        self.port = port
        self.server = None
        self.server_thread = None

    def start(self):
        """Start the server in a background thread."""
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)

        # Create a custom handler that correctly references the directory
        directory = self.directory  # Capture the directory for the closure

        class RaptorHandler(SimpleHTTPRequestHandler):
            # Set the directory class attribute
            #directory = directory  # This is the proper way to set it in Python 3.7+

            def __init__(self, *args, **kwargs):
                # In Python 3.7+, SimpleHTTPRequestHandler accepts directory as a parameter
                # For older versions, we need to set it before calling the parent constructor
                super().__init__(*args, **kwargs)

            def log_error(self, format, *args):
                print(f"Server error: {format % args}")

            def log_message(self, format, *args):
                # Override to provide more detailed logging
                print(f"Server: {format % args}")

        # Start server in a separate thread
        def run_server():
            try:
                self.server = HTTPServer(('localhost', self.port), RaptorHandler)
                print(f"Visualization server started at http://localhost:{self.port}")
                print(f"Serving files from: {self.directory}")

                # List files in directory to verify
                print("Files available for serving:")
                for file in os.listdir(self.directory):
                    print(f"  - {file}")

                self.server.serve_forever()
            except Exception as e:
                print(f"Server error: {str(e)}")

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def open_visualization(self, html_path):
        """Open visualization in browser."""
        if not os.path.exists(html_path):
            print(f"Error: Visualization file not found: {html_path}")
            return

        # Get the absolute path and check if the file exists
        abs_path = os.path.abspath(html_path)
        print(f"Visualization file path: {abs_path}")

        # Get the filename only for the URL
        filename = os.path.basename(html_path)
        url = f"http://localhost:{self.port}/{html_path}"

        print(f"Attempting to open URL: {url}")
        print(f"If the browser doesn't open, please manually navigate to the URL")

        try:
            webbrowser.open(url)
            print(f"Visualization opened in browser: {url}")
        except Exception as e:
            print(f"Error opening browser: {e}")
            print(f"Please visit: {url}")

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            print("Visualization server stopped")

def create_visualization_handler(server, output_dir):
    """
    Create a visualization handler function that uses the server.

    Args:
        server: VisualizationServer instance
        output_dir: Directory to save visualizations

    Returns:
        A handler function compatible with Raptor
    """

    def handler(hierarchy_data, output_path, namespace):
        # Create visualization
        html_path = RaptorVisualizationHandler.create_html_visualization(
            hierarchy_data,
            output_path,
            namespace
        )

        # Open in browser
        server.open_visualization(html_path)

        return html_path

    return handler


def direct_file_access(visualization_dir):
    """
    Print instructions for direct file access in case the server has issues.
    """
    abs_path = os.path.abspath(visualization_dir)
    print("\n===== VISUALIZATION ACCESS =====")
    print(f"If you can't access the visualization through the server,")
    print(f"you can open the HTML file directly in your browser:")
    print(f"Directory: {abs_path}")
    print(f"Look for files like: raptor_raptor_ns.html")
    print("===============================\n")