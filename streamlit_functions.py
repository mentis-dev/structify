import asyncio
import json
import os
import time
from datetime import datetime

import streamlit as st

# Import raptor components
from big_raptor.base import QueryModes
# Import the new StakeholderAnalyzer
from core.stakeholder_analyzer import StakeholderAnalyzer
from services.raptor_service import RaptorService
# Import our services
from services.supabase_service import SupabaseService


# Function to convert async to sync for Streamlit
def async_to_sync(coroutine):
    """Convert an async function to sync for Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine)
    loop.close()
    return result


# Function to initialize services automatically
def initialize_services():
    """Initialize Supabase and Raptor services automatically"""
    if st.session_state.supabase is None:
        try:
            # Get credentials from environment variables
            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")

            if not supabase_url or not supabase_key:
                st.sidebar.warning("⚠️ Missing Supabase credentials in environment. Please enter them manually.")
                return False

            # Set the environment variables for supabase_db module
            os.environ["SUPABASE_URL"] = supabase_url
            os.environ["SUPABASE_SERVICE_KEY"] = supabase_key

            # Initialize service
            from core.supabase_db import SUPABASE_URL, SUPABASE_SERVICE_KEY
            import core.supabase_db as supabase_db

            supabase_db.SUPABASE_URL = supabase_url
            supabase_db.SUPABASE_SERVICE_KEY = supabase_key

            # Initialize Supabase service
            st.session_state.supabase = SupabaseService(exclude_playgrounds=True)

            # Initialize Raptor service with default settings
            st.session_state.raptor_service = RaptorService(
                index_name="raptor_test",
                tree_depth=3,
                similarity_top_k=5,
                mode=QueryModes.collapsed,
                namespace="default"
            )

            return True
        except Exception as e:
            st.sidebar.error(f"Error auto-initializing services: {str(e)}")
            return False
    return True


# Function to load workspaces automatically
def load_workspaces():
    """Load workspaces automatically"""
    if st.session_state.supabase and not st.session_state.workspaces_loaded:
        try:
            with st.spinner("Loading workspaces..."):
                workspaces = st.session_state.supabase.get_workspaces()
                if workspaces:
                    # Update session state
                    st.session_state.workspaces = workspaces
                    st.session_state.workspace_options = {w["workspace_id"]: w["name"] for w in workspaces}
                    st.session_state.workspaces_loaded = True
                    return True
                else:
                    st.warning("No workspaces found.")
                    return False
        except Exception as e:
            st.error(f"Error loading workspaces: {str(e)}")
            return False
    return st.session_state.workspaces_loaded


# Function to load brains automatically
def load_brains():
    """Load brains for selected workspaces automatically"""
    if not st.session_state.selections["workspaces"]:
        return False

    if st.session_state.supabase and not st.session_state.brains_loaded:
        try:
            with st.spinner("Loading brains..."):
                all_brains = []
                for ws in st.session_state.selections["workspaces"]:
                    brains = st.session_state.supabase.get_brains_per_workspace(ws["workspace_id"])
                    if brains:
                        all_brains.extend(brains)

                if all_brains:
                    # Update session state
                    st.session_state.brains = all_brains
                    st.session_state.brain_options = {b["brain_id"]: b["name"] for b in all_brains}
                    st.session_state.brains_loaded = True
                    return True
                else:
                    st.warning("No brains found for selected workspaces.")
                    return False
        except Exception as e:
            st.error(f"Error loading brains: {str(e)}")
            return False
    return st.session_state.brains_loaded


# Function to load documents automatically
def load_documents():
    """Load documents for selected brains automatically"""
    if not st.session_state.selections["brains"]:
        return False

    if st.session_state.supabase and not st.session_state.documents_loaded_flag:
        try:
            with st.spinner("Loading documents..."):
                all_documents = []
                for brain in st.session_state.selections["brains"]:
                    documents = st.session_state.supabase.get_documents_per_brain(brain["brain_id"])
                    if documents:
                        all_documents.extend(documents)

                if all_documents:
                    # Update session state
                    st.session_state.documents = all_documents
                    st.session_state.document_options = {d["id"]: d["file_name"] for d in all_documents}
                    st.session_state.documents_loaded_flag = True
                    return True
                else:
                    st.warning("No documents found for selected brains.")
                    return False
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return False
    return st.session_state.documents_loaded_flag


# Function to run analysis on selected documents or texts using new StakeholderAnalyzer
def run_extraction_analysis():
    """Run stakeholder/factor/pain point extraction analysis on selected texts or documents"""

    # Check if we're in document mode or text mode
    if st.session_state.extraction_mode == "Selected Documents":
        # Document mode - require loaded documents
        if not st.session_state.selections["documents"]:
            st.warning("Please select documents first.")
            return False

        if not hasattr(st.session_state, "document_contents") or not st.session_state.document_contents:
            st.warning("Please load document content first.")
            return False
    else:
        # Text mode - require text input
        if not st.session_state.text_inputs:
            st.warning("Please add at least one text for analysis.")
            return False

    try:
        with st.spinner("Running extraction analysis..."):
            # Get the selected model from the sidebar
            model_name = st.session_state.get("extraction_model", "openai/gpt-4o")

            # Create a directory for extraction results if it doesn't exist
            output_dir = "extraction_output"
            os.makedirs(output_dir, exist_ok=True)

            # Initialize the StakeholderAnalyzer
            temperature = st.session_state.get("extraction_temperature", 0.1)
            analyzer = StakeholderAnalyzer(model=model_name, output_dir=output_dir, temperature=temperature)

            # Set up progress display
            progress_text = "Analyzing content..."
            my_bar = st.progress(0, text=progress_text)

            # Prepare texts for analysis
            texts_to_analyze = []

            if st.session_state.extraction_mode == "Selected Documents":
                # Get texts from loaded documents
                for i, doc in enumerate(st.session_state.selections["documents"]):
                    doc_id = doc["id"]
                    doc_name = doc["file_name"]

                    # Skip if content not loaded
                    if doc_id not in st.session_state.document_contents:
                        continue

                    # Get document content
                    content = st.session_state.document_contents[doc_id]

                    # If content is a list of chunks, combine them
                    if isinstance(content, list):
                        content = "\n\n".join(content)

                    # Limit content to avoid token limits (100k chars should be safe)
                    content = content[:100000]

                    texts_to_analyze.append({
                        "id": doc_id,
                        "name": doc_name,
                        "text": content
                    })
            else:
                # Get texts from manual input
                for i, text_input in enumerate(st.session_state.text_inputs):
                    text_id = f"text_{i + 1}_{int(time.time())}"
                    text_name = text_input.get("name", f"Text {i + 1}")

                    texts_to_analyze.append({
                        "id": text_id,
                        "name": text_name,
                        "text": text_input["text"]
                    })

            # Update progress bar
            total_texts = len(texts_to_analyze)

            if total_texts == 0:
                st.warning("No content available for analysis.")
                return False

            # Run the analysis with the new analyzer's multiple text function
            results = async_to_sync(analyzer.analyze_multiple_texts(
                texts=texts_to_analyze,
                verbose=False,
                aggregate=True
            ))

            # Update progress to 100%
            my_bar.progress(1.0, text="Processing completed!")

            # Store the results in session state
            st.session_state.extraction_results = results.get("individual_results", {})
            st.session_state.aggregated_results = results.get("aggregated_results", {})

            # Initialize feedback tracking for each type of extraction
            if "user_feedback" not in st.session_state:
                st.session_state.user_feedback = {
                    "stakeholders": {},
                    "factors": {},
                    "pain_points": {}
                }

            # Initialize feedback for new extraction results
            if "stakeholders" in st.session_state.aggregated_results:
                for i, stakeholder in enumerate(st.session_state.aggregated_results["stakeholders"]):
                    item_id = stakeholder.get("name", f"stakeholder_{i}")
                    # Only initialize if not already present
                    if item_id not in st.session_state.user_feedback["stakeholders"]:
                        st.session_state.user_feedback["stakeholders"][item_id] = {
                            "agrees": True,
                            "feedback": "",
                            "id": item_id,
                            "item_type": "stakeholder"
                        }

            if "factors" in st.session_state.aggregated_results:
                for i, factor in enumerate(st.session_state.aggregated_results["factors"]):
                    item_id = factor.get("Label", f"factor_{i}")
                    if item_id not in st.session_state.user_feedback["factors"]:
                        st.session_state.user_feedback["factors"][item_id] = {
                            "agrees": True,
                            "feedback": "",
                            "id": item_id,
                            "item_type": "factor"
                        }

            if "pain_points" in st.session_state.aggregated_results:
                for i, pain_point in enumerate(st.session_state.aggregated_results["pain_points"]):
                    item_id = pain_point.get("id", f"pain_point_{i}")
                    if item_id not in st.session_state.user_feedback["pain_points"]:
                        st.session_state.user_feedback["pain_points"][item_id] = {
                            "agrees": True,
                            "feedback": "",
                            "id": item_id,
                            "item_type": "pain_point"
                        }

            return True

    except Exception as e:
        st.error(f"Error running extraction analysis: {str(e)}")
        return False


# Add this function near your other utility functions
def initialize_feedback_for_results(result):
    """Initialize user feedback tracking for extraction results"""
    if "user_feedback" not in st.session_state:
        st.session_state.user_feedback = {
            "stakeholders": {},
            "factors": {},
            "pain_points": {}
        }

    # Initialize feedback for stakeholders
    if "stakeholders" in result:
        for i, stakeholder in enumerate(result["stakeholders"]):
            item_id = stakeholder.get("name", f"stakeholder_{i}")
            if item_id not in st.session_state.user_feedback["stakeholders"]:
                st.session_state.user_feedback["stakeholders"][item_id] = {
                    "agrees": True,
                    "feedback": "",
                    "id": item_id,
                    "item_type": "stakeholder"
                }

    # Initialize feedback for factors
    if "factors" in result:
        for i, factor in enumerate(result["factors"]):
            item_id = factor.get("Label", f"factor_{i}")
            if item_id not in st.session_state.user_feedback["factors"]:
                st.session_state.user_feedback["factors"][item_id] = {
                    "agrees": True,
                    "feedback": "",
                    "id": item_id,
                    "item_type": "factor"
                }

    # Initialize feedback for pain points
    if "pain_points" in result:
        for i, pain_point in enumerate(result["pain_points"]):
            item_id = pain_point.get("id", f"pain_point_{i}")
            if item_id not in st.session_state.user_feedback["pain_points"]:
                st.session_state.user_feedback["pain_points"][item_id] = {
                    "agrees": True,
                    "feedback": "",
                    "id": item_id,
                    "item_type": "pain_point"
                }

# Function to load extraction results for display
def load_extraction_results():
    """Load extraction results from files"""
    output_dir = "extraction_output"
    if not os.path.exists(output_dir):
        return None

    results = {}

    # Look for doc_*_analysis.json or text_*_analysis.json files
    for filename in os.listdir(output_dir):
        if (filename.startswith("doc_") or filename.startswith("text_")) and filename.endswith("_analysis.json"):
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    doc_id = data.get("document_info", {}).get("id", filename.split("_analysis.json")[0])
                    results[doc_id] = data
            except Exception as e:
                st.warning(f"Error loading extraction file {filename}: {str(e)}")

    # Look for aggregated_analysis_*.json file
    aggregated_analysis = None
    for filename in os.listdir(output_dir):
        if filename.startswith("aggregated_analysis_") and filename.endswith(".json"):
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, "r") as f:
                    aggregated_analysis = json.load(f)
            except Exception as e:
                st.warning(f"Error loading aggregated analysis file {filename}: {str(e)}")

    return {
        "individual_results": results,
        "aggregated_results": aggregated_analysis
    }


# Function to save user feedback
def save_user_feedback():
    """Save all user feedback to a JSON file"""
    if not hasattr(st.session_state, "user_feedback"):
        return

    try:
        # Create a directory for feedback if it doesn't exist
        feedback_dir = "user_feedback"
        os.makedirs(feedback_dir, exist_ok=True)

        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the feedback
        filename = f"{feedback_dir}/feedback_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(st.session_state.user_feedback, f, indent=2)

        return filename
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
        return None


# Function to render the feedback interface for a specific item
def render_feedback_interface(item_id, item_type):
    """Render feedback interface for a specific extraction item"""
    # Get the feedback object
    feedback_obj = st.session_state.user_feedback[item_type].get(item_id)
    if not feedback_obj:
        return False, ""

    # Create a unique key for this item
    key_base = f"{item_type}_{item_id}".replace(" ", "_").lower()

    # Create columns for agreement and feedback
    col1, col2 = st.columns([1, 5])

    with col1:
        # Agreement checkbox
        agrees = st.checkbox(
            "Agree",
            value=feedback_obj.get("agrees", True),
            key=f"agrees_{key_base}"
        )

        # Update feedback object
        feedback_obj["agrees"] = agrees

    with col2:
        # Only show the feedback text field if the user disagrees
        if not agrees:
            feedback = st.text_area(
                "Please explain why you disagree:",
                value=feedback_obj.get("feedback", ""),
                key=f"feedback_{key_base}",
                height=100
            )

            # Update feedback object
            feedback_obj["feedback"] = feedback
        else:
            feedback = feedback_obj.get("feedback", "")

    # Return current state
    return agrees, feedback


# Function to save RAPTOR analysis results
def save_raptor_analysis_results(results, level=None, cluster_id=None):
    """
    Save RAPTOR analysis results to disk.

    Args:
        results: The analysis results to save
        level: The RAPTOR hierarchy level (for all-clusters analysis)
        cluster_id: The specific cluster ID (for single-cluster analysis)

    Returns:
        Path to the saved file
    """
    import os
    import json
    from datetime import datetime

    # Create directory for storing results
    output_dir = "raptor_extraction"
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine filename based on analysis type
    if cluster_id:
        # Single cluster analysis
        filename = f"{output_dir}/cluster_{cluster_id}_analysis_{timestamp}.json"
    elif level is not None:
        # All clusters at level
        filename = f"{output_dir}/level_{level}_analysis_{timestamp}.json"
    else:
        # Generic filename if no specific type
        filename = f"{output_dir}/raptor_analysis_{timestamp}.json"

    # Save metadata about the analysis
    results_with_metadata = {
        "analysis_type": "cluster" if cluster_id else "level",
        "level": level,
        "cluster_id": cluster_id,
        "timestamp": timestamp,
        "results": results
    }

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2)

    return filename


def list_saved_raptor_analyses():
    """
    List all saved RAPTOR analysis results.

    Returns:
        List of dictionaries with information about saved analyses
    """
    import os
    import json
    import glob
    from datetime import datetime

    output_dir = "raptor_extraction"

    # Check if directory exists
    if not os.path.exists(output_dir):
        return []

    # Find all JSON files in the directory
    pattern = os.path.join(output_dir, "*.json")
    json_files = glob.glob(pattern)

    saved_analyses = []

    for file_path in json_files:
        try:
            # Read the file to extract metadata
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract metadata
            analysis_type = data.get("analysis_type", "unknown")
            level = data.get("level")
            cluster_id = data.get("cluster_id")
            timestamp_str = data.get("timestamp")

            # Format a descriptive name
            if analysis_type == "cluster" and cluster_id:
                description = f"Cluster {cluster_id} Analysis"
            elif analysis_type == "level" and level is not None:
                description = f"Level {level} Analysis"
            else:
                description = "RAPTOR Analysis"

            # Add timestamp if available
            if timestamp_str:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    formatted_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    description += f" ({formatted_date})"
                except:
                    # Use the raw timestamp if parsing fails
                    description += f" ({timestamp_str})"

            # Check if it has the expected data
            has_aggregated = "results" in data and "aggregated_results" in data["results"]
            has_individual = "results" in data and "individual_results" in data["results"]

            # For a direct analysis result (not wrapped)
            if not has_aggregated and not has_individual:
                has_stakeholders = "stakeholders" in data
                has_factors = "factors" in data
                has_pain_points = "pain_points" in data

                if has_stakeholders or has_factors or has_pain_points:
                    # This is likely a direct analysis result
                    description += " (Direct)"

            saved_analyses.append({
                "file_path": file_path,
                "description": description,
                "analysis_type": analysis_type,
                "level": level,
                "cluster_id": cluster_id,
                "timestamp": timestamp_str,
                "filename": os.path.basename(file_path)
            })

        except Exception as e:
            # Skip files that can't be properly parsed
            print(f"Error parsing {file_path}: {str(e)}")
            continue

    # Sort by timestamp (newest first)
    try:
        saved_analyses.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    except:
        pass

    return saved_analyses


def load_raptor_analysis(file_path):
    """
    Load a saved RAPTOR analysis from disk.

    Args:
        file_path: Path to the saved analysis file

    Returns:
        Tuple of (individual_results, aggregated_results) or None if error
    """
    import json

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if this is a metadata-wrapped analysis
        if "results" in data:
            wrapped_results = data["results"]

            # Check for different result formats
            if "individual_results" in wrapped_results and "aggregated_results" in wrapped_results:
                # Standard format with both individual and aggregated results
                return wrapped_results["individual_results"], wrapped_results["aggregated_results"]
            elif "aggregated_results" in wrapped_results:
                # Only aggregated results
                return {}, wrapped_results["aggregated_results"]
            else:
                # Assume the results themselves are the aggregated results
                return {}, wrapped_results
        else:
            # This might be a direct analysis result
            has_stakeholders = "stakeholders" in data
            has_factors = "factors" in data
            has_pain_points = "pain_points" in data

            if has_stakeholders or has_factors or has_pain_points:
                # This is likely a direct analysis result
                return {}, data

            # As a fallback, return the data directly
            return {}, data

    except Exception as e:
        st.error(f"Error loading analysis: {str(e)}")
        return None


# Add these functions to your Streamlit app to manage FAISS indexes

def list_saved_faiss_indexes():
    """
    List all saved FAISS indexes.

    Returns:
        List of dictionaries containing information about saved indexes
    """
    import os
    import glob
    import json
    from datetime import datetime

    # Directory where FAISS indexes are stored
    index_folder = "faiss_indexes"

    if not os.path.exists(index_folder):
        return []

    # Find all index files
    index_files = glob.glob(os.path.join(index_folder, "*.faiss"))
    metadata_files = glob.glob(os.path.join(index_folder, "*_metadata.pkl"))

    # Extract index names from file paths
    index_names = [os.path.basename(f).replace(".faiss", "") for f in index_files]

    # Create a list of index information
    indexes = []

    for index_name in index_names:
        index_info = {
            "name": index_name,
            "file_path": os.path.join(index_folder, f"{index_name}.faiss"),
            "metadata_path": os.path.join(index_folder, f"{index_name}_metadata.pkl"),
        }

        # Check if there's a metadata JSON file with additional info
        info_path = os.path.join(index_folder, f"{index_name}_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)

                # Add info to index data
                index_info.update(info)

                # Format creation date
                if "created_at" in info:
                    try:
                        created_at = datetime.fromisoformat(info["created_at"])
                        index_info["created_at_formatted"] = created_at.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        index_info["created_at_formatted"] = info["created_at"]
            except:
                # If JSON is invalid, just continue
                pass

        # Check if metadata file exists
        index_info["has_metadata"] = os.path.exists(index_info["metadata_path"])

        indexes.append(index_info)

    # Sort by name
    indexes.sort(key=lambda x: x["name"])

    return indexes


def save_index_metadata(index_name, metadata):
    """
    Save additional metadata about a FAISS index.

    Args:
        index_name: Name of the index
        metadata: Dictionary containing metadata
    """
    import os
    import json
    from datetime import datetime

    # Directory where FAISS indexes are stored
    index_folder = "faiss_indexes"
    os.makedirs(index_folder, exist_ok=True)

    # Add timestamp
    metadata["created_at"] = datetime.now().isoformat()

    # Save metadata as JSON
    info_path = os.path.join(index_folder, f"{index_name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return info_path


def load_faiss_index(raptor_service, index_name, namespace="default"):
    """
    Load a FAISS index into the Raptor service.

    Args:
        raptor_service: The RaptorService instance
        index_name: Name of the index to load
        namespace: Namespace to use

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use the existing manager in the raptor service
        manager = raptor_service.manager

        # Check if the index exists
        index_path = os.path.join("faiss_indexes", f"{index_name}.faiss")
        metadata_path = os.path.join("faiss_indexes", f"{index_name}_metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return False

        # Update the index name in the manager
        manager.index_name = index_name

        # Load the index into the manager
        if namespace not in manager.indexes:
            manager._load_or_create_namespace(namespace)

        return True
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
        return False


# Add this to Tab 2, in the RAPTOR Processing section:
def add_index_management_ui():
    """Add UI elements to save and load FAISS indexes"""
    st.subheader("FAISS Index Management")

    # Create columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Save Current Index")

        # Only enable if an index exists
        if st.session_state.indexed_status != "Not indexed":
            # Custom index name input
            custom_index_name = st.text_input(
                "Index Name",
                value=st.session_state.raptor_service.index_name,
                help="Give this index a descriptive name"
            )

            # Additional metadata
            description = st.text_area(
                "Description (optional)",
                placeholder="Describe the contents of this index..."
            )

            # Save button
            if st.button("Save Current Index"):
                with st.spinner("Saving index..."):
                    try:
                        # Force persistence to ensure latest data is saved
                        st.session_state.raptor_service.persist()

                        # Save additional metadata
                        metadata = {
                            "description": description,
                            "original_name": st.session_state.raptor_service.index_name,
                            "namespace": st.session_state.raptor_service.namespace,
                            "document_count": len(st.session_state.langchain_docs) if hasattr(st.session_state,
                                                                                              "langchain_docs") else 0,
                            "user": os.getenv("USER", "unknown")
                        }

                        save_index_metadata(custom_index_name, metadata)

                        st.success(f"Index saved as '{custom_index_name}'")
                    except Exception as e:
                        st.error(f"Error saving index: {str(e)}")
        else:
            st.info("Index documents first to enable saving")

    with col2:
        st.subheader("Load Existing Index")

        # List available indexes
        indexes = list_saved_faiss_indexes()

        if not indexes:
            st.info("No saved indexes found")
        else:
            # Create a selection widget
            options = [
                f"{idx['name']}{' - ' + idx.get('description', '')[:30] + '...' if idx.get('description') else ''}"
                for idx in indexes]

            selected_index = st.selectbox(
                "Select Index to Load",
                options=range(len(options)),
                format_func=lambda i: options[i]
            )

            # Show index details
            selected = indexes[selected_index]

            st.markdown(f"**Name:** {selected['name']}")
            if "description" in selected:
                st.markdown(f"**Description:** {selected['description']}")
            if "created_at_formatted" in selected:
                st.markdown(f"**Created:** {selected['created_at_formatted']}")
            if "document_count" in selected:
                st.markdown(f"**Documents:** {selected['document_count']}")

            # Namespace selection
            namespace = st.text_input("Namespace", value="default")

            # Load button
            if st.button("Load Selected Index"):
                with st.spinner(f"Loading index '{selected['name']}'..."):
                    try:
                        success = load_faiss_index(
                            st.session_state.raptor_service,
                            selected['name'],
                            namespace=namespace
                        )

                        if success:
                            # Update session state to reflect the loaded index
                            st.session_state.raptor_service.index_name = selected['name']
                            st.session_state.raptor_service.namespace = namespace
                            st.session_state.indexed_status = f"Loaded from {selected['name']}"

                            # Attempt to load hierarchy data for visualization
                            try:
                                st.session_state.hierarchy_data = generate_hierarchy_data(
                                    st.session_state.raptor_service.retriever,
                                    namespace=namespace
                                )
                            except:
                                pass  # Ignore errors in generating hierarchy data

                            st.success(f"Successfully loaded index '{selected['name']}'")

                            # Force a rerun to update all state
                            st.experimental_rerun()
                        else:
                            st.error(f"Failed to load index '{selected['name']}'")
                    except Exception as e:
                        st.error(f"Error loading index: {str(e)}")


# Add a function to automatically save index metadata when processing with Raptor
def save_index_after_processing(index_name, document_count):
    """Save metadata after processing documents with Raptor"""
    metadata = {
        "description": f"Index created with {document_count} documents",
        "original_name": index_name,
        "namespace": st.session_state.raptor_service.namespace,
        "document_count": document_count,
        "user": os.getenv("USER", "unknown")
    }

    save_index_metadata(index_name, metadata)
