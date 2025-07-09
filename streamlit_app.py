import json
import os
from datetime import datetime
import glob

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import mammoth  # For DOCX processing

# Import raptor components
from big_raptor.base import QueryModes
from big_raptor.raptor_visualization import display_raptor_visualization, generate_hierarchy_data
# Import the StakeholderAnalyzer and new StakeholderValidator
from core.raptor_stakeholder_analyzer import RaptorStakeholderAnalyzer, analyze_raptor_level, analyze_raptor_cluster
from core.stakeholder_validator import validate_stakeholders  # NEW IMPORT
from services.document_service import convert_supabase_to_langchain
from services.raptor_service import RaptorService
# Import our services
from services.supabase_service import SupabaseService
from core.feedback_system import (FeedbackLearningAgent, 
    FeedbackStorage, 
    initialize_feedback_system,
    process_streamlit_feedback,
    reprocess_with_feedback
)

import sqlite3  # Add this if not already imported
import asyncio  # Add this if not already imported

from streamlit_functions import *
# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Raptor FAISS Retriever",
    page_icon="🦖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state
if "supabase" not in st.session_state:
    st.session_state.supabase = None

if "raptor_service" not in st.session_state:
    st.session_state.raptor_service = None

if "selections" not in st.session_state:
    st.session_state.selections = {
        "workspaces": [],
        "brains": [],
        "documents": []
    }

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "indexed_status" not in st.session_state:
    st.session_state.indexed_status = "Not indexed"

if "query_results" not in st.session_state:
    st.session_state.query_results = None

# Session state for data loading
if "workspaces_loaded" not in st.session_state:
    st.session_state.workspaces_loaded = False

if "brains_loaded" not in st.session_state:
    st.session_state.brains_loaded = False

if "documents_loaded_flag" not in st.session_state:
    st.session_state.documents_loaded_flag = False

# Session state for data storage
if "workspaces" not in st.session_state:
    st.session_state.workspaces = []

if "workspace_options" not in st.session_state:
    st.session_state.workspace_options = {}

if "brains" not in st.session_state:
    st.session_state.brains = []

if "brain_options" not in st.session_state:
    st.session_state.brain_options = {}

if "documents" not in st.session_state:
    st.session_state.documents = []

if "document_options" not in st.session_state:
    st.session_state.document_options = {}

if "use_chunks" not in st.session_state:
    st.session_state.use_chunks = True

# Session state for extraction results
if "extraction_results" not in st.session_state:
    st.session_state.extraction_results = {}

if "aggregated_results" not in st.session_state:
    st.session_state.aggregated_results = None

if "active_extraction_tab" not in st.session_state:
    st.session_state.active_extraction_tab = "Stakeholders"

if "extraction_model" not in st.session_state:
    st.session_state.extraction_model = "openai/gpt-4.1"  # Changed default to gpt-4.1

# New session state for text extraction mode
if "extraction_mode" not in st.session_state:
    st.session_state.extraction_mode = "Selected Documents"

if "text_inputs" not in st.session_state:
    st.session_state.text_inputs = []
    
# New session state for user feedback
if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = {
        "stakeholders": {},
        "factors": {},
        "pain_points": {}
    }

# ADD THESE NEW SESSION STATE VARIABLES right after the existing ones:
if "feedback_agent" not in st.session_state:
    st.session_state.feedback_agent = None

if "feedback_enabled" not in st.session_state:
    st.session_state.feedback_enabled = True

if "feedback_system_initialized" not in st.session_state:
    st.session_state.feedback_system_initialized = False
    
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = True

# NEW: Session state for stakeholder validation
if "show_validation_options" not in st.session_state:
    st.session_state.show_validation_options = True  # Enable by default

if "validation_results" not in st.session_state:
    st.session_state.validation_results = None

if "validation_in_progress" not in st.session_state:
    st.session_state.validation_in_progress = False

if "original_results" not in st.session_state:
    st.session_state.original_results = None

# NEW: Session state for DOCX directory processing
if "docx_directory" not in st.session_state:
    st.session_state.docx_directory = ""

if "docx_files" not in st.session_state:
    st.session_state.docx_files = []

if "selected_docx_files" not in st.session_state:
    st.session_state.selected_docx_files = []

if "docx_contents" not in st.session_state:
    st.session_state.docx_contents = {}

if "docx_loaded" not in st.session_state:
    st.session_state.docx_loaded = False

# Helper functions for DOCX processing
def load_docx_files_from_directory(directory_path):
    """Load all DOCX files from the specified directory"""
    try:
        if not os.path.exists(directory_path):
            st.error(f"Directory does not exist: {directory_path}")
            return []
        
        # Find all DOCX files in the directory
        docx_pattern = os.path.join(directory_path, "*.docx")
        docx_files = glob.glob(docx_pattern)
        
        # Filter out temporary files (starting with ~$)
        docx_files = [f for f in docx_files if not os.path.basename(f).startswith("~$")]
        
        # Create file info list
        file_info = []
        for file_path in docx_files:
            file_stat = os.stat(file_path)
            file_info.append({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return file_info
    except Exception as e:
        st.error(f"Error loading DOCX files: {str(e)}")
        return []

def extract_text_from_docx(file_path):
    """Extract text content from a DOCX file using mammoth"""
    try:
        with open(file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return result.value
    except Exception as e:
        st.error(f"Error extracting text from {file_path}: {str(e)}")
        return None

def process_docx_files_for_extraction(selected_files):
    """Process selected DOCX files and extract text content"""
    docx_contents = {}
    
    for file_info in selected_files:
        file_path = file_info["file_path"]
        file_name = file_info["file_name"]
        
        # Extract text content
        text_content = extract_text_from_docx(file_path)
        
        if text_content:
            docx_contents[file_name] = {
                "content": text_content,
                "file_path": file_path,
                "file_info": file_info
            }
    
    return docx_contents

def run_docx_extraction_analysis_sync():
    """Run extraction analysis on selected DOCX files using existing patterns"""
    try:
        if not st.session_state.selected_docx_files:
            st.error("No DOCX files selected for analysis.")
            return False
        
        # Process DOCX files to extract text content
        with st.spinner("Extracting text from DOCX files..."):
            docx_contents = process_docx_files_for_extraction(st.session_state.selected_docx_files)
        
        if not docx_contents:
            st.error("No text content could be extracted from the selected files.")
            return False
        
        st.session_state.docx_contents = docx_contents
        st.success(f"Successfully extracted text from {len(docx_contents)} DOCX files")
        
        # Create temporary text inputs that mimic the existing text input workflow
        temp_text_inputs = []
        for file_name, file_data in docx_contents.items():
            temp_text_inputs.append({
                "name": file_name,
                "text": file_data["content"]
            })
        
        # Temporarily replace text inputs with DOCX content
        original_text_inputs = st.session_state.text_inputs
        original_extraction_mode = st.session_state.extraction_mode
        
        # Set up for text processing
        st.session_state.text_inputs = temp_text_inputs
        st.session_state.extraction_mode = "Direct Text Input"
        
        # Use the existing run_extraction_analysis function
        success = run_extraction_analysis()
        
        # Restore original state
        st.session_state.text_inputs = original_text_inputs
        st.session_state.extraction_mode = original_extraction_mode
        
        return success
            
    except Exception as e:
        st.error(f"Error in DOCX extraction analysis: {str(e)}")
        # Restore original state in case of error
        if 'original_text_inputs' in locals():
            st.session_state.text_inputs = original_text_inputs
        if 'original_extraction_mode' in locals():
            st.session_state.extraction_mode = original_extraction_mode
        return False

# Callback for workspace selection changes
def on_workspace_change():
    # Clear downstream selections when workspaces change
    st.session_state.selections["brains"] = []
    st.session_state.selections["documents"] = []
    st.session_state.brains_loaded = False
    st.session_state.documents_loaded_flag = False
    st.session_state.documents_loaded = False


# Callback for brain selection changes
def on_brain_change():
    # Clear downstream selections when brains change
    st.session_state.selections["documents"] = []
    st.session_state.documents_loaded_flag = False
    st.session_state.documents_loaded = False


# Header
st.title("🦖 Raptor FAISS Retriever & Analysis Tool")
st.markdown("### Connect Supabase documents to Raptor via FAISS")

# Auto-initialize services
services_initialized = initialize_services()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Show current status
    if services_initialized:
        st.success("✅ Services auto-initialized")
    else:
        st.warning("⚠️ Manual initialization required")

    # Manual configuration (collapsed by default if auto-initialized)
    with st.expander("Manual Configuration", expanded=not services_initialized):
        # Supabase credentials
        supabase_url = st.text_input("Supabase URL", value=os.getenv("SUPABASE_URL", ""))
        supabase_key = st.text_input("Supabase Key", value=os.getenv("SUPABASE_SERVICE_KEY", ""), type="password")

        # Playground filter option
        exclude_playgrounds = st.checkbox("Exclude Playground Workspaces", value=True)

        st.divider()

        # Raptor configuration
        st.subheader("Raptor Configuration")

        # Index name
        index_name = st.text_input("Index Name", "raptor_test")

        # Namespace
        namespace = st.text_input("Namespace", "default")

        # Tree depth
        tree_depth = st.slider("Tree Depth", 1, 5, 3)

        # Retrieval mode
        mode_options = ["collapsed", "tree_traversal"]
        selected_mode = st.selectbox("Default Retrieval Mode", mode_options)
        mode = QueryModes.collapsed if selected_mode == "collapsed" else QueryModes.tree_traversal

        # Top K results
        top_k = st.slider("Top K Results", 1, 20, 5)

        # Apply configuration button
        if st.button("Initialize Services Manually"):
            # Initialize Supabase service
            try:
                if not supabase_url or not supabase_key:
                    st.error("Supabase URL and Key are required")
                else:
                    with st.spinner("Initializing Supabase..."):
                        # Set the environment variables for supabase_db module
                        os.environ["SUPABASE_URL"] = supabase_url
                        os.environ["SUPABASE_SERVICE_KEY"] = supabase_key

                        # Initialize service
                        from core.supabase_db import SUPABASE_URL, SUPABASE_SERVICE_KEY
                        import core.supabase_db as supabase_db

                        supabase_db.SUPABASE_URL = supabase_url
                        supabase_db.SUPABASE_SERVICE_KEY = supabase_key

                        st.session_state.supabase = SupabaseService(exclude_playgrounds=exclude_playgrounds)
                        st.success("Supabase initialized successfully!")

                        # Initialize Raptor service
                        with st.spinner("Initializing Raptor..."):
                            st.session_state.raptor_service = RaptorService(
                                index_name=index_name,
                                tree_depth=tree_depth,
                                similarity_top_k=top_k,
                                mode=mode,
                                namespace=namespace
                            )
                            st.success("Raptor initialized successfully!")

                        # Reset data loading flags
                        st.session_state.workspaces_loaded = False
                        st.session_state.brains_loaded = False
                        st.session_state.documents_loaded_flag = False
                        st.session_state.documents_loaded = False
            except Exception as e:
                st.error(f"Error initializing services: {str(e)}")

    # Document processing options
    st.subheader("Document Processing")

    # Option to use pre-chunked documents
    st.session_state.use_chunks = st.checkbox("Use Pre-chunked Documents", value=True,
                                            help="Use the pre-chunked documents from Supabase instead of treating each document as a whole")

    # Model selection for extraction
    st.session_state.extraction_model = st.selectbox(
        "Extraction Model",
        ["openai/gpt-4o", "openai/gpt-4.1", "anthropic/claude-3-opus", "anthropic/claude-3-sonnet"],
        index=1,  # Changed to index 1 to make gpt-4.1 the default
        help="Select the AI model for stakeholder extraction"
    )

    # Temperature control for extraction
    st.session_state.extraction_temperature = st.slider(
        "Model Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower values (0.0-0.3) for more deterministic results, higher values (0.7-1.0) for more creative responses"
    )

    # Feedback configuration
    st.subheader("Feedback Options")

    # Toggle for showing feedback interface
    st.session_state.show_feedback = st.checkbox(
        "Enable User Feedback Interface",
        value=st.session_state.show_feedback,
        help="Show checkboxes and comment fields for providing feedback on extraction results"
    )

    # Validation options
    st.subheader("Validation Options")
    
    # Show validation options toggle
    st.session_state.show_validation_options = st.checkbox(
        "Enable Stakeholder Validation",
        value=st.session_state.show_validation_options,
        help="Automatically validate stakeholders by finding duplicates and normalizing categories"
    )

    # NEW: Feedback Learning System
    st.subheader("Feedback Learning System")
    
    # Initialize feedback system
    if not st.session_state.feedback_system_initialized:
        if st.button("Initialize Feedback System"):
            try:
                st.session_state.feedback_agent = initialize_feedback_system()
                st.session_state.feedback_system_initialized = True
                st.success("✅ Feedback system initialized!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Failed to initialize feedback system: {e}")
    else:
        st.success("✅ Feedback system active")
        
        # Show feedback stats if system is initialized
        if st.session_state.feedback_agent:
            try:
                stats = st.session_state.feedback_agent.feedback_storage.get_feedback_stats()
                
                if stats['total_feedback'] > 0:
                    st.write("**Learning Statistics:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Feedback", stats['total_feedback'])
                        st.metric("Accuracy Rate", f"{stats['accuracy_rate']:.1%}")
                    with col2:
                        st.metric("Positive", stats['positive_feedback'])
                        st.metric("Negative", stats['negative_feedback'])
                else:
                    st.info("No feedback data yet")
            except Exception as e:
                st.warning(f"Could not load feedback stats: {e}")
    
    # Toggle for using feedback in classifications
    st.session_state.feedback_enabled = st.checkbox(
        "Use Learned Feedback",
        value=st.session_state.feedback_enabled,
        help="When enabled, the system will use previous feedback to improve classifications",
        disabled=not st.session_state.feedback_system_initialized
    )

    # NEW: DOCX Directory Configuration
    st.subheader("DOCX Directory Processing")
    
    # Directory path input
    default_docx_dir = os.path.join(os.getcwd(), "docx_files")
    st.session_state.docx_directory = st.text_input(
        "DOCX Directory Path",
        value=st.session_state.docx_directory or default_docx_dir,
        help="Path to directory containing DOCX files to process"
    )
    
    # Show directory info
    if st.session_state.docx_directory:
        if os.path.exists(st.session_state.docx_directory):
            docx_count = len(glob.glob(os.path.join(st.session_state.docx_directory, "*.docx")))
            st.info(f"📁 Found {docx_count} DOCX files in directory")
        else:
            st.warning(f"📁 Directory does not exist: {st.session_state.docx_directory}")

    # Extraction schema configuration
    with st.expander("Extraction Schema", expanded=False):
        st.code("""
{
  "type": "object",
  "properties": {
    "stakeholders": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the stakeholder organization or individual."
          },
          "category": {
            "type": "string",
            "description": "The category of the stakeholder (Regulator, Supplier, Consumer, Competitor, Partner, Influencer, Internal)."
          },
          "role": {
            "type": "string",
            "description": "Brief description of the stakeholder's role or function."
          },
          "confidence": {
            "type": "number",
            "description": "Confidence score (0-100%) for the classification."
          },
          "hierarchy_level": {
            "type": "string",
            "description": "Level in the ecosystem hierarchy (Macro, Meso, Micro)."
          }
        },
        "required": ["name", "category", "role", "confidence", "hierarchy_level"]
      }
    },
    "factors": { ... },
    "pain_points": { ... }
  }
}
        """, language="json")

    # Display current service settings
    if st.session_state.supabase is not None and st.session_state.raptor_service is not None:
        st.subheader("Current Settings")
        st.info(f"""
        **Supabase:** Connected ✅
        **Raptor Index:** {st.session_state.raptor_service.index_name}
        **Namespace:** {st.session_state.raptor_service.namespace}
        **Tree Depth:** {st.session_state.raptor_service.retriever.tree_depth}
        **Mode:** {st.session_state.raptor_service.retriever.mode}
        **Top K:** {st.session_state.raptor_service.retriever.similarity_top_k}
        **Using Chunks:** {"Yes" if st.session_state.use_chunks else "No"}
        **Extraction Mode:** {st.session_state.extraction_mode}
        **Extraction Model:** {st.session_state.extraction_model}
        **Feedback Interface:** {"Enabled" if st.session_state.show_feedback else "Disabled"}
        **Validation Interface:** {"Enabled" if st.session_state.show_validation_options else "Disabled"}
        **DOCX Directory:** {st.session_state.docx_directory}
        """)

    st.subheader("Extraction Mode Selection")

    # Extend extraction mode to include RAPTOR and DOCX options
    st.session_state.extraction_mode = st.radio(
        "Choose Analysis Source",
        ["Selected Documents", "Direct Text Input", "RAPTOR Clusters", "DOCX Directory"],
        index=["Selected Documents", "Direct Text Input", "RAPTOR Clusters", "DOCX Directory"].index(st.session_state.extraction_mode) if st.session_state.extraction_mode in ["Selected Documents", "Direct Text Input", "RAPTOR Clusters", "DOCX Directory"] else 0,
        help="Choose the source for text analysis"
    )
    
    # Show current selection
    if st.session_state.extraction_mode == "DOCX Directory":
        if st.session_state.docx_loaded:
            st.success("✅ DOCX mode - Content loaded and ready")
        else:
            st.warning("⚠️ DOCX mode - Load content in DOCX Directory tab first")
    elif st.session_state.extraction_mode == "Selected Documents":
        if st.session_state.documents_loaded:
            st.success("✅ Supabase mode - Documents loaded and ready")
        else:
            st.warning("⚠️ Supabase mode - Load documents in Document Selection tab first")

    st.subheader("RAPTOR Analysis Options")

    # Only show RAPTOR options when RAPTOR mode is selected
    if st.session_state.extraction_mode == "RAPTOR Clusters":
        # Initialize RAPTOR analyzer if needed
        if st.session_state.raptor_service is not None:
            # Load available levels
            if "available_levels" not in st.session_state:
                st.session_state.available_levels = []

            try:
                analyzer = RaptorStakeholderAnalyzer(
                    raptor_service=st.session_state.raptor_service,
                    namespace=st.session_state.raptor_service.namespace,
                    verbose=False
                )
                st.session_state.available_levels = analyzer.get_level_options()
            except Exception as e:
                st.sidebar.error(f"Error loading RAPTOR levels: {str(e)}")
                st.session_state.available_levels = []

            # RAPTOR analysis type
            raptor_analysis_type = st.sidebar.radio(
                "RAPTOR Analysis Type",
                ["All Clusters at Level", "Single Cluster"],
                help="Choose to analyze all clusters at a level or a specific cluster"
            )

            # Level selection
            if st.session_state.available_levels:
                level_options = [f"Level {level}" for level in st.session_state.available_levels]
                selected_level_idx = st.sidebar.selectbox(
                    "Select RAPTOR Level",
                    options=range(len(level_options)),
                    format_func=lambda i: level_options[i],
                    index=min(1, len(level_options) - 1) if len(level_options) > 1 else 0
                )
                st.session_state.raptor_level = st.session_state.available_levels[selected_level_idx]

                # Cluster selection (only for "Single Cluster" mode)
                if raptor_analysis_type == "Single Cluster":
                    # Load clusters for the selected level
                    try:
                        clusters = async_to_sync(analyzer.get_clusters_at_level(st.session_state.raptor_level))

                        if clusters:
                            cluster_options = [f"Cluster {i + 1}: {c['content_preview'][:30]}..." for i, c in
                                            enumerate(clusters)]
                            selected_cluster_idx = st.sidebar.selectbox(
                                "Select Cluster",
                                options=range(len(cluster_options)),
                                format_func=lambda i: cluster_options[i]
                            )
                            st.session_state.selected_cluster = clusters[selected_cluster_idx]["id"]
                            st.session_state.available_clusters = clusters
                        else:
                            st.sidebar.warning(f"No clusters found at level {st.session_state.raptor_level}")
                            st.session_state.selected_cluster = None
                            st.session_state.available_clusters = []
                    except Exception as e:
                        st.sidebar.error(f"Error loading clusters: {str(e)}")
                        st.session_state.selected_cluster = None
                        st.session_state.available_clusters = []
            else:
                st.sidebar.warning("No RAPTOR levels available. Please index documents first.")
                st.session_state.raptor_level = 1
        else:
            st.sidebar.warning("RAPTOR service not initialized. Please initialize in the sidebar.")

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Document Selection",
    "DOCX Directory",  # NEW TAB
    "Raptor Processing", 
    "Raptor Visualization",
    "Extraction Analysis",
    "Feedback Management"
])


# Tab 1: Document Selection (keeping the existing code unchanged)
with tab1:
    st.header("Select Documents from Supabase")

    if st.session_state.supabase is None:
        st.warning("Please initialize Supabase in the sidebar first.")
    else:
        # Display playground filter status
        if st.session_state.supabase.exclude_playgrounds:
            st.info("Playground workspaces are being filtered out. Change this in the sidebar if needed.")

        # Auto-load workspaces
        workspaces_loaded = load_workspaces()

        # Workspace section
        st.subheader("1. Select Workspaces")

        # Load workspaces button - only show if auto-load failed
        if not workspaces_loaded:
            if st.button("Load Workspaces"):
                load_workspaces()

        # Display workspace selection if data is loaded
        if st.session_state.workspaces_loaded and st.session_state.workspaces:
            # Create a DataFrame for display
            df = pd.DataFrame(st.session_state.workspaces)
            st.dataframe(df[["workspace_id", "name"]])

            # Workspace selection - always present after loading
            selected_workspace_ids = st.multiselect(
                "Select Workspaces",
                options=list(st.session_state.workspace_options.keys()),
                format_func=lambda x: st.session_state.workspace_options.get(x, x),
                on_change=on_workspace_change,
                key="workspace_multiselect"
            )

            # Update selections when user changes them
            if selected_workspace_ids:
                st.session_state.selections["workspaces"] = [
                    w for w in st.session_state.workspaces if w["workspace_id"] in selected_workspace_ids
                ]
                # Show success message
                st.success(f"Selected {len(selected_workspace_ids)} workspaces")
            else:
                # Clear selections if nothing is selected
                st.session_state.selections["workspaces"] = []

        # Brain section
        st.subheader("2. Select Brains")

        # Only show if workspaces are selected
        if not st.session_state.selections["workspaces"]:
            st.warning("Please select workspaces first.")
        else:
            # Auto-load brains
            brains_loaded = load_brains()

            # Load brains button - only show if auto-load failed
            if not brains_loaded:
                if st.button("Load Brains"):
                    load_brains()

            # Display brain selection if data is loaded
            if st.session_state.brains_loaded and st.session_state.brains:
                # Create a DataFrame for display
                df = pd.DataFrame(st.session_state.brains)
                st.dataframe(df[["brain_id", "name"]])

                # Brain selection - always present after loading
                selected_brain_ids = st.multiselect(
                    "Select Brains",
                    options=list(st.session_state.brain_options.keys()),
                    format_func=lambda x: st.session_state.brain_options.get(x, x),
                    on_change=on_brain_change,
                    key="brain_multiselect"
                )

                # Update selections when user changes them
                if selected_brain_ids:
                    st.session_state.selections["brains"] = [
                        b for b in st.session_state.brains if b["brain_id"] in selected_brain_ids
                    ]
                    # Show success message
                    st.success(f"Selected {len(selected_brain_ids)} brains")
                else:
                    # Clear selections if nothing is selected
                    st.session_state.selections["brains"] = []

        # Document section
        st.subheader("3. Select Documents")

        # Only show if brains are selected
        if not st.session_state.selections["brains"]:
            st.warning("Please select brains first.")
        else:
            # Auto-load documents
            documents_loaded = load_documents()

            # Load documents button - only show if auto-load failed
            if not documents_loaded:
                if st.button("Load Documents"):
                    load_documents()

            # Display document selection if data is loaded
            if st.session_state.documents_loaded_flag and st.session_state.documents:
                # Create a DataFrame for display
                df = pd.DataFrame(st.session_state.documents)
                st.dataframe(df[["id", "file_name", "brain_id"]])

                # Document selection - always present after loading
                selected_doc_ids = st.multiselect(
                    "Select Documents",
                    options=list(st.session_state.document_options.keys()),
                    format_func=lambda x: st.session_state.document_options.get(x, x),
                    key="document_multiselect"
                )

                # Update selections when user changes them
                if selected_doc_ids:
                    st.session_state.selections["documents"] = [
                        d for d in st.session_state.documents if d["id"] in selected_doc_ids
                    ]
                    # Show success message
                    st.success(f"Selected {len(selected_doc_ids)} documents")
                else:
                    # Clear selections if nothing is selected
                    st.session_state.selections["documents"] = []

        # Document content section
        st.subheader("4. Load Document Content")

        # Only show if documents are selected
        if not st.session_state.selections["documents"]:
            st.warning("Please select documents first.")
        else:
            # Load document content button
            if st.button("Load Document Content"):
                with st.spinner("Loading document content..."):
                    document_ids = [d["id"] for d in st.session_state.selections["documents"]]

                    # Get document chunks from Supabase
                    # Note: We always request chunks, even if we'll later combine them
                    document_contents = {}
                    for doc_id in document_ids:
                        content = st.session_state.supabase.get_document_content(doc_id, return_chunks=True)
                        if content:
                            document_contents[doc_id] = content

                    if document_contents:
                        st.session_state.document_contents = document_contents
                        st.session_state.documents_loaded = True

                        # Count total chunks
                        total_chunks = sum(len(chunks) for chunks in document_contents.values())
                        st.success(
                            f"Successfully loaded content for {len(document_contents)} documents ({total_chunks} total chunks)")
                    else:
                        st.error("Failed to load document content.")

            # Display sample content if loaded
            if hasattr(st.session_state, "document_contents") and st.session_state.document_contents:
                first_doc_id = list(st.session_state.document_contents.keys())[0]
                first_doc_content = st.session_state.document_contents[first_doc_id]

                if isinstance(first_doc_content, list):
                    # Show first chunk and indicate total chunks
                    sample = first_doc_content[0] if first_doc_content else ""
                    st.text_area(f"Sample content (Chunk 1/{len(first_doc_content)})",
                                 sample[:1000] + ("..." if len(sample) > 1000 else ""),
                                 height=200)
                else:
                    st.text_area("Sample content",
                                 first_doc_content[:1000] + ("..." if len(first_doc_content) > 1000 else ""),
                                 height=200)

# Tab 2: NEW DOCX Directory Processing
with tab2:
    st.header("DOCX Directory Processing")
    st.markdown("Load and process DOCX files from a local directory for stakeholder analysis.")

    # Directory path configuration
    st.subheader("1. Configure Directory Path")
    
    # Directory path input with browse option
    col1, col2 = st.columns([3, 1])
    
    with col1:
        directory_path = st.text_input(
            "DOCX Directory Path",
            value=st.session_state.docx_directory,
            help="Enter the full path to the directory containing DOCX files"
        )
    
    with col2:
        if st.button("📁 Browse", help="Use current working directory + docx_files"):
            default_path = os.path.join(os.getcwd(), "docx_files")
            st.session_state.docx_directory = default_path
            st.rerun()
    
    # Update session state
    if directory_path != st.session_state.docx_directory:
        st.session_state.docx_directory = directory_path
        st.session_state.docx_files = []  # Reset file list when directory changes
    
    # Directory validation and file discovery
    if st.session_state.docx_directory:
        if os.path.exists(st.session_state.docx_directory):
            st.success(f"✅ Directory found: {st.session_state.docx_directory}")
            
            # Load DOCX files from directory
            if st.button("🔍 Scan for DOCX Files") or not st.session_state.docx_files:
                with st.spinner("Scanning directory for DOCX files..."):
                    docx_files = load_docx_files_from_directory(st.session_state.docx_directory)
                    st.session_state.docx_files = docx_files
                    
                    if docx_files:
                        st.success(f"Found {len(docx_files)} DOCX files")
                    else:
                        st.warning("No DOCX files found in the specified directory")
        else:
            st.error(f"❌ Directory not found: {st.session_state.docx_directory}")
            st.info("Please check the path or create the directory first.")
    
    # File selection section
    if st.session_state.docx_files:
        st.subheader("2. Select DOCX Files")
        
        # Display files in a dataframe
        df = pd.DataFrame(st.session_state.docx_files)
        st.dataframe(df[["file_name", "size_mb", "modified"]], use_container_width=True)
        
        # File selection
        file_options = {i: f"{file_info['file_name']} ({file_info['size_mb']} MB)" 
                       for i, file_info in enumerate(st.session_state.docx_files)}
        
        selected_indices = st.multiselect(
            "Select DOCX files to process",
            options=list(file_options.keys()),
            format_func=lambda i: file_options[i],
            default=list(range(len(st.session_state.docx_files)))  # Select all by default
        )
        
        # Update selected files
        st.session_state.selected_docx_files = [
            st.session_state.docx_files[i] for i in selected_indices
        ]
        
        if st.session_state.selected_docx_files:
            st.success(f"Selected {len(st.session_state.selected_docx_files)} files for processing")
            
            # Show total size
            total_size = sum(f["size_mb"] for f in st.session_state.selected_docx_files)
            st.info(f"Total size: {total_size:.2f} MB")
        else:
            st.warning("No files selected")
    
    # Text extraction and content loading section (similar to Document Selection tab)
    if st.session_state.selected_docx_files:
        st.subheader("3. Load DOCX Content")
        
        # Load content button (similar to "Load Document Content" in tab 1)
        if st.button("📄 Load DOCX Content"):
            with st.spinner("Extracting text from DOCX files..."):
                try:
                    docx_contents = process_docx_files_for_extraction(st.session_state.selected_docx_files)
                    
                    if docx_contents:
                        st.session_state.docx_contents = docx_contents
                        st.session_state.docx_loaded = True
                        
                        # Show success message similar to document loading
                        st.success(f"Successfully extracted text from {len(docx_contents)} DOCX files")
                        
                        # Show text statistics
                        total_chars = sum(len(content["content"]) for content in docx_contents.values())
                        st.info(f"Total text extracted: {total_chars:,} characters")
                    else:
                        st.error("Failed to extract text from DOCX files.")
                        st.session_state.docx_loaded = False
                        
                except Exception as e:
                    st.error(f"Error extracting text: {str(e)}")
                    st.session_state.docx_loaded = False
        
        # Display sample content if loaded (similar to Document Selection tab)
        if st.session_state.docx_loaded and st.session_state.docx_contents:
            st.subheader("4. Content Preview")
            
            # Show first file content as preview
            first_file_name = list(st.session_state.docx_contents.keys())[0]
            first_file_content = st.session_state.docx_contents[first_file_name]["content"]
            
            # Display preview
            preview_text = first_file_content[:1000] + ("..." if len(first_file_content) > 1000 else "")
            st.text_area(
                f"Sample content from {first_file_name}",
                preview_text,
                height=200,
                help="This is a preview of the extracted text content"
            )
            
            # Show summary statistics for all files
            with st.expander("📊 Content Summary", expanded=False):
                summary_data = []
                for file_name, file_data in st.session_state.docx_contents.items():
                    content = file_data["content"]
                    word_count = len(content.split())
                    char_count = len(content)
                    
                    summary_data.append({
                        "File": file_name,
                        "Characters": f"{char_count:,}",
                        "Words": f"{word_count:,}",
                        "Size (MB)": file_data["file_info"]["size_mb"]
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            
            # Ready for analysis message
            st.success("✅ DOCX content loaded successfully!")
            st.info("📋 Go to the **'Extraction Analysis'** tab to analyze the loaded DOCX content.")
            
            # Show quick action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Go to Analysis Tab"):
                    st.session_state.extraction_mode = "DOCX Directory"
                    st.info("Switch to the 'Extraction Analysis' tab to run the analysis.")
            
            with col2:
                if st.button("🔄 Clear Loaded Content"):
                    st.session_state.docx_contents = {}
                    st.session_state.docx_loaded = False
                    st.rerun()
    
    # Status summary section
    if st.session_state.selected_docx_files or st.session_state.docx_loaded:
        st.divider()
        st.subheader("📋 Current Status")
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            files_selected = len(st.session_state.selected_docx_files) if st.session_state.selected_docx_files else 0
            st.metric("Files Selected", files_selected)
        
        with col2:
            content_loaded = "✅ Yes" if st.session_state.docx_loaded else "❌ No"
            st.metric("Content Loaded", content_loaded)
        
        with col3:
            extraction_mode = "✅ Ready" if st.session_state.extraction_mode == "DOCX Directory" else "📋 Set Mode"
            st.metric("Analysis Mode", extraction_mode)
        
        # Next steps guidance
        if st.session_state.docx_loaded:
            st.success("🎯 **Next Step:** Go to the 'Extraction Analysis' tab and click 'Run Extraction Analysis'")
        elif st.session_state.selected_docx_files:
            st.info("🎯 **Next Step:** Click 'Load DOCX Content' above to extract text from your selected files")
        else:
            st.info("🎯 **Next Step:** Select DOCX files to process")
    
    # Help section
    with st.expander("ℹ️ How to Use DOCX Directory Processing", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Set Directory Path**: Enter the path to your folder containing DOCX files
        2. **Scan for Files**: Click 'Scan for DOCX Files' to find all DOCX files in the directory
        3. **Select Files**: Choose which DOCX files you want to analyze
        4. **Load Content**: Click 'Load DOCX Content' to extract text from the selected files
        5. **Run Analysis**: Go to the 'Extraction Analysis' tab and click 'Run Extraction Analysis'
        
        ### What happens during analysis:
        - Each DOCX file is analyzed individually for stakeholders, factors, and pain points
        - Results from all files are then aggregated together
        - Automatic validation and feedback learning (if enabled) are applied
        - You can provide feedback on the results to improve future analyses
        
        ### Supported features:
        - ✅ Individual file analysis with aggregation
        - ✅ Stakeholder validation and deduplication  
        - ✅ Feedback learning system
        - ✅ Result visualization and export
        """)


# Tab 3: Raptor Processing (keeping existing code)
with tab3:
    st.header("Process Documents with Raptor")

    if not st.session_state.documents_loaded:
        st.warning("Please load document content in the Document Selection tab first.")
    elif st.session_state.raptor_service is None:
        st.warning("Please initialize Raptor in the sidebar first.")
    else:
        # Convert documents to LangChain format
        if st.button("Convert Documents to LangChain Format"):
            with st.spinner("Converting documents..."):
                try:
                    # Get document information
                    document_infos = st.session_state.selections["documents"]
                    document_contents = st.session_state.document_contents
                    use_chunks = st.session_state.use_chunks

                    # Convert to LangChain documents
                    langchain_docs = convert_supabase_to_langchain(
                        document_infos,
                        document_contents,
                        use_chunks=use_chunks
                    )

                    # Store in session state
                    st.session_state.langchain_docs = langchain_docs
                    st.success(f"Successfully converted to {len(langchain_docs)} LangChain documents")

                    # Show chunking info
                    if use_chunks:
                        st.info("Using pre-chunked documents from Supabase")
                    else:
                        st.info("Using complete documents (no pre-chunking)")

                except Exception as e:
                    st.error(f"Error converting documents: {str(e)}")

        # Process with Raptor
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Raptor Indexing")

            fresh_start = st.checkbox("Fresh Start (Clear existing index)", value=True)

            if st.button("Process with Raptor"):
                if not hasattr(st.session_state, "langchain_docs"):
                    st.warning("Please convert documents to LangChain format first.")
                else:
                    with st.spinner("Processing documents with Raptor..."):
                        try:
                            langchain_docs = st.session_state.langchain_docs
                            namespace = st.session_state.raptor_service.namespace

                            # Process documents with Raptor
                            process_status = async_to_sync(
                                st.session_state.raptor_service.index_documents(
                                    langchain_docs,
                                    namespace=namespace,
                                    fresh_start=fresh_start
                                )
                            )

                            # Update indexing status
                            st.session_state.indexed_status = f"Indexed {len(langchain_docs)} documents"
                            st.success(f"Successfully processed {len(langchain_docs)} documents with Raptor")
                            save_index_after_processing(st.session_state.raptor_service.index_name, len(langchain_docs))

                            # Generate visualization data automatically
                            with st.spinner("Generating visualization data..."):
                                try:
                                    # Generate and store hierarchy data
                                    st.session_state.hierarchy_data = generate_hierarchy_data(
                                        st.session_state.raptor_service.retriever,
                                        namespace=namespace
                                    )
                                    st.success("✅ Visualization data generated successfully")
                                except Exception as e:
                                    st.warning(f"⚠️ Visualization data could not be generated: {str(e)}")

                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")

        with col2:
            st.subheader("Indexing Status")

            # Display indexing status
            st.info(f"Status: {st.session_state.indexed_status}")

            # Display FAISS index info if available
            if st.session_state.raptor_service is not None:
                index_name = st.session_state.raptor_service.index_name
                namespace = st.session_state.raptor_service.namespace
                tree_depth = st.session_state.raptor_service.retriever.tree_depth

                st.text(f"Index Name: {index_name}")
                st.text(f"Namespace: {namespace}")
                st.text(f"Tree Depth: {tree_depth}")

                # Display information about chunking strategy
                if hasattr(st.session_state, "langchain_docs"):
                    st.text(f"Document Count: {len(st.session_state.langchain_docs)}")
                    st.text(f"Using Pre-chunked: {st.session_state.use_chunks}")

    # After indexing is complete, show a visualization preview
    # In the "Process with Raptor" button handler, after successful processing
    if st.session_state.indexed_status != "Not indexed":
        if st.button("View Document Hierarchy"):
            # Display the visualization in an expander
            with st.expander("Document Hierarchy Visualization", expanded=True):
                # Display a compact version of the visualization
                if "hierarchy_data" in st.session_state:
                    # Import visualization functions
                    from big_raptor.raptor_visualization import (
                        create_tree_visualization,
                        create_level_pie_chart
                    )

                    # Create a simplified preview
                    col1, col2 = st.columns(2)

                    with col1:
                        # Tree visualization
                        tree_fig = create_tree_visualization(st.session_state.hierarchy_data)
                        st.plotly_chart(tree_fig, use_container_width=True, key="preview_tree")

                    with col2:
                        # Pie chart visualization
                        pie_fig = create_level_pie_chart(st.session_state.hierarchy_data)
                        st.plotly_chart(pie_fig, use_container_width=True, key="preview_pie")

                    st.info("For more detailed visualizations, go to the 'Raptor Visualization' tab.")
                else:
                    # If hierarchy data isn't available, generate it
                    try:
                        hierarchy_data = generate_hierarchy_data(
                            st.session_state.raptor_service.retriever,
                            namespace=st.session_state.raptor_service.namespace
                        )
                        st.session_state.hierarchy_data = hierarchy_data

                        # Now display the visualization with unique keys
                        from big_raptor.raptor_visualization import (
                            create_tree_visualization,
                            create_level_pie_chart
                        )

                        # Create a simplified preview
                        col1, col2 = st.columns(2)

                        with col1:
                            # Tree visualization
                            tree_fig = create_tree_visualization(hierarchy_data)
                            st.plotly_chart(tree_fig, use_container_width=True, key="generated_preview_tree")

                        with col2:
                            # Pie chart visualization
                            pie_fig = create_level_pie_chart(hierarchy_data)
                            st.plotly_chart(pie_fig, use_container_width=True, key="generated_preview_pie")

                        st.info("For more detailed visualizations, go to the 'Raptor Visualization' tab.")

                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")

# Tab 4: Raptor Visualization
with tab4:
    if st.session_state.indexed_status == "Not indexed":
        st.warning("Please index documents with Raptor in the Processing tab first.")
        st.info("No RAPTOR index loaded. You can load an existing index:")

        # List available indexes
        indexes = list_saved_faiss_indexes()

        if not indexes:
            st.warning("No saved indexes found. Please index documents in the Raptor Processing tab first.")
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

    elif st.session_state.raptor_service is None:
        st.warning("Please initialize Raptor in the sidebar first.")
    else:
        # Get current namespace
        namespace = st.session_state.raptor_service.namespace

        # Display the visualization
        display_raptor_visualization(
            st.session_state.raptor_service,
            namespace=namespace
        )

# Tab 5: Extraction Analysis
with tab5:
    st.header("Document Extraction Analysis")

    # Show different content based on the extraction mode
    if st.session_state.extraction_mode == "RAPTOR Clusters":
        # RAPTOR cluster mode
        if st.session_state.raptor_service is None:
            st.warning("Please initialize RAPTOR service in the sidebar first.")
        else:
            # Display RAPTOR-specific information
            st.subheader("RAPTOR Cluster Analysis")

            # Show information about the current RAPTOR selection
            if hasattr(st.session_state, "raptor_level"):
                level = st.session_state.raptor_level
                st.info(f"Selected RAPTOR hierarchy level: {level}")

                # Show selected cluster info if in Single Cluster mode
                if st.session_state.extraction_mode == "RAPTOR Clusters" and hasattr(st.session_state,
                                                                                    "selected_cluster") and st.session_state.selected_cluster:
                    st.info(f"Selected cluster: {st.session_state.selected_cluster}")

                # Information about how RAPTOR analysis works
                with st.expander("How RAPTOR Analysis Works", expanded=False):
                    st.markdown("""
                    ### RAPTOR Cluster Analysis Process

                    1. The system finds all level 0 documents that are descendants of your selected cluster(s)
                    2. All document content is combined and sent to the StakeholderAnalyzer
                    3. Results include stakeholders, factors, and pain points found in the combined content
                    4. This lets you analyze content at different levels of the document hierarchy
                    """)

    elif st.session_state.extraction_mode == "Selected Documents":
        # Document mode
        if not st.session_state.documents_loaded:
            st.warning("Please load document content in the Document Selection tab first.")
        else:
            # Show document selection summary
            st.subheader("Selected Documents for Analysis")

            if st.session_state.selections["documents"]:
                doc_names = [doc["file_name"] for doc in st.session_state.selections["documents"]]
                st.write(f"You have selected {len(doc_names)} documents:")
                st.write(", ".join(doc_names))
            else:
                st.warning("No documents selected for analysis.")

    elif st.session_state.extraction_mode == "DOCX Directory":
        # DOCX directory mode
        st.subheader("DOCX Directory Analysis")
        
        if st.session_state.selected_docx_files:
            st.write(f"Processing {len(st.session_state.selected_docx_files)} DOCX files from:")
            st.code(st.session_state.docx_directory)
            
            # Show selected files
            file_names = [f["file_name"] for f in st.session_state.selected_docx_files]
            st.write("**Selected files:**")
            for name in file_names:
                st.write(f"• {name}")
                
            # Show processing status
            if st.session_state.docx_contents:
                st.success(f"✅ Text extracted from {len(st.session_state.docx_contents)} files")
            else:
                st.info("Ready to process. Use the 'DOCX Directory' tab to run analysis.")
        else:
            st.warning("No DOCX files selected. Please configure and select files in the 'DOCX Directory' tab.")

    else:
        # Direct text input mode
        st.subheader("Direct Text Input for Analysis")

        # Show text entry interface
        with st.expander("Add New Text for Analysis", expanded=len(st.session_state.text_inputs) == 0):
            text_name = st.text_input("Text Name/Title", value="")
            text_content = st.text_area("Text Content", height=300)

            if st.button("Add Text"):
                if not text_content.strip():
                    st.error("Please enter some text content.")
                else:
                    # Add to session state
                    new_text = {
                        "name": text_name if text_name else f"Text {len(st.session_state.text_inputs) + 1}",
                        "text": text_content
                    }
                    st.session_state.text_inputs.append(new_text)
                    st.success(f"Added text: {new_text['name']}")

        # Show existing text inputs
        if st.session_state.text_inputs:
            st.subheader(f"Texts for Analysis ({len(st.session_state.text_inputs)})")

            for i, text_input in enumerate(st.session_state.text_inputs):
                with st.expander(f"{i + 1}. {text_input['name']}", expanded=False):
                    st.text_area(f"Content", text_input["text"][:1000] +
                                ("..." if len(text_input["text"]) > 1000 else ""),
                                height=150,
                                key=f"text_view_{i}")

                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state.text_inputs.pop(i)
                        st.rerun()

            # Clear all button
            if st.button("Clear All Texts"):
                st.session_state.text_inputs = []
                st.rerun()
        else:
            st.info("Add texts using the form above to analyze them.")

    # Extraction Process Section - common for all modes
    st.divider()
    st.subheader("Run Extraction Analysis")

    col1, col2 = st.columns(2)
    
    with col1:
        # Option to run extraction - adapt based on mode
        if st.button("Run Extraction Analysis"):
            if st.session_state.extraction_mode == "RAPTOR Clusters":
                # RAPTOR-based extraction
                if st.session_state.raptor_service is None:
                    st.warning("Raptor service is not initialized")
                elif st.session_state.indexed_status == "Not indexed":
                    st.warning(
                        "No documents have been indexed in RAPTOR. Please index documents in the Raptor Processing tab or load an existing index first.")
                if st.session_state.raptor_service is None:
                    st.warning("Raptor service is not initialized")
                else:
                    try:
                        with st.spinner("Running RAPTOR extraction analysis..."):
                            # Get the selected model from the sidebar
                            model_name = st.session_state.get("extraction_model", "openai/gpt-4o")
                            temperature = st.session_state.get("extraction_temperature", 0.1)

                            # Create a directory for extraction results if it doesn't exist
                            output_dir = "raptor_extraction"
                            os.makedirs(output_dir, exist_ok=True)

                            # Determine analysis type
                            raptor_analysis_type = "Single Cluster" if hasattr(st.session_state,
                                                                              "selected_cluster") and st.session_state.selected_cluster else "All Clusters at Level"

                            if raptor_analysis_type == "Single Cluster" and st.session_state.selected_cluster:
                                # Analyze the selected cluster
                                result = async_to_sync(analyze_raptor_cluster(
                                    raptor_service=st.session_state.raptor_service,
                                    cluster_id=st.session_state.selected_cluster,
                                    cluster_name=f"Cluster {st.session_state.selected_cluster}",
                                    model=model_name,
                                    temperature=temperature,
                                    output_dir=output_dir,
                                    namespace=st.session_state.raptor_service.namespace,
                                    verbose=True
                                ))

                                if result.get('error'):
                                    st.error(f"Error analyzing cluster: {result.get('error')}")
                                else:
                                    # Store original results
                                    st.session_state.extraction_results = {st.session_state.selected_cluster: result}
                                    st.session_state.original_results = result.copy()
                                    
                                    # Automatically run validation if enabled
                                    if st.session_state.show_validation_options and "stakeholders" in result:
                                        with st.spinner("Validating stakeholders..."):
                                            try:
                                                # Run validation
                                                validation_results = async_to_sync(validate_stakeholders(
                                                    stakeholders=result["stakeholders"],
                                                    model=model_name,
                                                    output_dir=output_dir,
                                                    verbose=True
                                                ))
                                                
                                                # Store validation results
                                                st.session_state.validation_results = validation_results
                                                
                                                # Update stakeholders in the aggregated results
                                                result["stakeholders"] = validation_results.get("stakeholders", [])
                                                result["stakeholder_validation"] = {
                                                    'original_count': validation_results.get("original_stakeholders", 0),
                                                    'validated_count': validation_results.get("validated_stakeholders", 0),
                                                    'duplicates_merged': validation_results.get("duplicates_merged", 0),
                                                    'category_changes': validation_results.get("category_changes", 0)
                                                }
                                            except Exception as e:
                                                st.warning(f"Stakeholder validation failed: {str(e)}")
                                    
                                    # Store final results
                                    st.session_state.aggregated_results = result

                                    # Initialize feedback for the results
                                    initialize_feedback_for_results(result)

                                    st.success(
                                        f"✅ Extraction analysis completed for cluster {st.session_state.selected_cluster}")
                                    try:
                                        # Auto-save the results
                                        saved_file = save_raptor_analysis_results(
                                            results=result,
                                            cluster_id=st.session_state.selected_cluster
                                        )
                                        st.info(f"Results automatically saved to: {saved_file}")
                                    except Exception as e:
                                        st.warning(f"Could not auto-save results: {str(e)}")
                            else:
                                # Analyze all clusters at the specified level
                                level = st.session_state.raptor_level

                                # Run the analysis
                                results = async_to_sync(analyze_raptor_level(
                                    raptor_service=st.session_state.raptor_service,
                                    level=level,
                                    model=model_name,
                                    temperature=temperature,
                                    output_dir=output_dir,
                                    namespace=st.session_state.raptor_service.namespace,
                                    verbose=True,
                                    aggregate=True
                                ))

                                if results.get('error'):
                                    st.error(f"Error analyzing level {level}: {results.get('error')}")
                                else:
                                    # Store original results
                                    st.session_state.extraction_results = results.get('individual_results', {})
                                    aggregated_results = results.get('aggregated_results', {})
                                    st.session_state.original_results = aggregated_results.copy()
                                    
                                    # Automatically run validation if enabled
                                    if st.session_state.show_validation_options and "stakeholders" in aggregated_results:
                                        with st.spinner("Validating stakeholders..."):
                                            try:
                                                # Run validation
                                                validation_results = async_to_sync(validate_stakeholders(
                                                    stakeholders=aggregated_results["stakeholders"],
                                                    model=model_name,
                                                    output_dir=output_dir,
                                                    verbose=True
                                                ))
                                                
                                                # Store validation results
                                                st.session_state.validation_results = validation_results
                                                
                                                # Update stakeholders in the aggregated results
                                                aggregated_results["stakeholders"] = validation_results.get("stakeholders", [])
                                                aggregated_results["stakeholder_validation"] = {
                                                    'original_count': validation_results.get("original_stakeholders", 0),
                                                    'validated_count': validation_results.get("validated_stakeholders", 0),
                                                    'duplicates_merged': validation_results.get("duplicates_merged", 0),
                                                    'category_changes': validation_results.get("category_changes", 0)
                                                }
                                            except Exception as e:
                                                st.warning(f"Stakeholder validation failed: {str(e)}")
                                    
                                    # Store final results
                                    st.session_state.aggregated_results = aggregated_results

                                    # Initialize feedback for the results
                                    initialize_feedback_for_results(st.session_state.aggregated_results)

                                    st.success(
                                        f"✅ Extraction analysis completed for {results.get('clusters_analyzed', 0)} clusters at level {level}")
                                    try:
                                        # Auto-save the results
                                        saved_file = save_raptor_analysis_results(
                                            results=results,
                                            level=level
                                        )
                                        st.info(f"Results automatically saved to: {saved_file}")
                                    except Exception as e:
                                        st.warning(f"Could not auto-save results: {str(e)}")
                    except Exception as e:
                        st.error(f"Error running RAPTOR extraction analysis: {str(e)}")
            elif st.session_state.extraction_mode == "DOCX Directory":
                # DOCX directory processing
                if not st.session_state.docx_loaded:
                    st.warning("No DOCX content loaded. Please go to the 'DOCX Directory' tab and load DOCX content first.")
                elif not st.session_state.selected_docx_files:
                    st.warning("No DOCX files selected. Please go to the 'DOCX Directory' tab and select files first.")
                else:
                    # Create output directory
                    os.makedirs("docx_extraction", exist_ok=True)
                    
                    # Run the analysis using the loaded DOCX content
                    success = run_docx_extraction_analysis_sync()
                    
                    if success:
                        # Store original results
                        if st.session_state.aggregated_results:
                            st.session_state.original_results = st.session_state.aggregated_results.copy()
                            
                            # Automatically run validation if enabled
                            if (st.session_state.show_validation_options and 
                                "stakeholders" in st.session_state.aggregated_results):
                                
                                with st.spinner("Validating stakeholders..."):
                                    try:
                                        # Get the selected model from the sidebar
                                        model_name = st.session_state.get("extraction_model", "openai/gpt-4o")
                                        
                                        # Run validation
                                        validation_results = async_to_sync(validate_stakeholders(
                                            stakeholders=st.session_state.aggregated_results["stakeholders"],
                                            model=model_name,
                                            output_dir="docx_extraction",
                                            verbose=True
                                        ))
                                        
                                        # Store validation results
                                        st.session_state.validation_results = validation_results
                                        
                                        # Update stakeholders in the aggregated results
                                        st.session_state.aggregated_results["stakeholders"] = validation_results.get("stakeholders", [])
                                        st.session_state.aggregated_results["stakeholder_validation"] = {
                                            'original_count': validation_results.get("original_stakeholders", 0),
                                            'validated_count': validation_results.get("validated_stakeholders", 0),
                                            'duplicates_merged': validation_results.get("duplicates_merged", 0),
                                            'category_changes': validation_results.get("category_changes", 0)
                                        }
                                    except Exception as e:
                                        st.warning(f"Stakeholder validation failed: {str(e)}")
                        
                        # Apply learned feedback if available
                        if st.session_state.feedback_agent and st.session_state.feedback_enabled:
                            try:
                                with st.spinner("Applying learned feedback improvements..."):
                                    improved_results = async_to_sync(reprocess_with_feedback(
                                        st.session_state.feedback_agent,
                                        st.session_state.aggregated_results
                                    ))
                                    st.session_state.aggregated_results = improved_results
                                    st.info("✨ Classifications improved using previous feedback")
                            except Exception as e:
                                st.warning(f"Could not apply feedback improvements: {e}")
                        
                        st.success("✅ DOCX extraction analysis completed successfully")
                        st.session_state.active_extraction_tab = "Stakeholders"
                        
                        # Try to auto-save results
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_file = f"docx_extraction/docx_analysis_{timestamp}.json"
                            
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump({
                                    "individual_results": st.session_state.extraction_results,
                                    "aggregated_results": st.session_state.aggregated_results,
                                    "processing_info": {
                                        "timestamp": timestamp,
                                        "files_processed": [f["file_name"] for f in st.session_state.selected_docx_files],
                                        "directory": st.session_state.docx_directory,
                                        "model": st.session_state.extraction_model,
                                        "validation_enabled": st.session_state.show_validation_options
                                    }
                                }, f, indent=2, ensure_ascii=False)
                            
                            st.info(f"📄 Results saved to: {output_file}")
                        except Exception as e:
                            st.warning(f"Could not auto-save results: {str(e)}")
                            
                    else:
                        st.error("❌ DOCX extraction analysis failed")
            else:
                ## Original code for document/text-based extraction
                #success = run_extraction_analysis()
                # Enhanced extraction with feedback learning
                if st.session_state.feedback_agent and st.session_state.feedback_enabled:
                    st.info("🧠 Using learned feedback to improve classifications...")
                
                # Original code for document/text-based extraction
                success = run_extraction_analysis()
                
                if success:
                    # Store original results
                    if st.session_state.aggregated_results:
                        st.session_state.original_results = st.session_state.aggregated_results.copy()       
                        with st.spinner("Validating stakeholders..."):
                            try:
                                # Get the selected model from the sidebar
                                model_name = st.session_state.get("extraction_model", "openai/gpt-4o")
                                
                                # Run validation
                                validation_results = async_to_sync(validate_stakeholders(
                                    stakeholders=st.session_state.aggregated_results["stakeholders"],
                                    model=model_name,
                                    output_dir="raptor_extraction",
                                    verbose=True
                                ))
                                
                                # Store validation results
                                st.session_state.validation_results = validation_results
                                
                                # Update stakeholders in the aggregated results
                                st.session_state.aggregated_results["stakeholders"] = validation_results.get("stakeholders", [])
                                st.session_state.aggregated_results["stakeholder_validation"] = {
                                    'original_count': validation_results.get("original_stakeholders", 0),
                                    'validated_count': validation_results.get("validated_stakeholders", 0),
                                    'duplicates_merged': validation_results.get("duplicates_merged", 0),
                                    'category_changes': validation_results.get("category_changes", 0)
                                }
                            except Exception as e:
                                st.warning(f"Stakeholder validation failed: {str(e)}")
                    
                    # Apply learned feedback if available
                    if st.session_state.feedback_agent and st.session_state.feedback_enabled:
                        try:
                            with st.spinner("Applying learned feedback improvements..."):
                                improved_results = async_to_sync(reprocess_with_feedback(
                                    st.session_state.feedback_agent,
                                    st.session_state.aggregated_results
                                ))
                                st.session_state.aggregated_results = improved_results
                                st.info("✨ Classifications improved using previous feedback")
                        except Exception as e:
                            st.warning(f"Could not apply feedback improvements: {e}")
                    
                    st.success("✅ Extraction analysis completed successfully")
                    st.session_state.active_extraction_tab = "Stakeholders"
                else:
                    st.error("❌ Extraction analysis failed")

                if success:
                    # Store original results
                    if st.session_state.aggregated_results:
                        st.session_state.original_results = st.session_state.aggregated_results.copy()
                    
                    # Automatically run validation if enabled
                    if (st.session_state.show_validation_options and 
                        st.session_state.aggregated_results and 
                        "stakeholders" in st.session_state.aggregated_results):
                        
                        with st.spinner("Validating stakeholders..."):
                            try:
                                # Get the selected model from the sidebar
                                model_name = st.session_state.get("extraction_model", "openai/gpt-4o")
                                
                                # Run validation
                                validation_results = async_to_sync(validate_stakeholders(
                                    stakeholders=st.session_state.aggregated_results["stakeholders"],
                                    model=model_name,
                                    output_dir="raptor_extraction",
                                    verbose=True
                                ))
                                
                                # Store validation results
                                st.session_state.validation_results = validation_results
                                
                                # Update stakeholders in the aggregated results
                                st.session_state.aggregated_results["stakeholders"] = validation_results.get("stakeholders", [])
                                st.session_state.aggregated_results["stakeholder_validation"] = {
                                    'original_count': validation_results.get("original_stakeholders", 0),
                                    'validated_count': validation_results.get("validated_stakeholders", 0),
                                    'duplicates_merged': validation_results.get("duplicates_merged", 0),
                                    'category_changes': validation_results.get("category_changes", 0)
                                }
                            except Exception as e:
                                st.warning(f"Stakeholder validation failed: {str(e)}")
                    
                    st.success("✅ Extraction analysis completed successfully")
                    st.session_state.active_extraction_tab = "Stakeholders"
                else:
                    st.error("❌ Extraction analysis failed")

    with col2:
        # Create a container for load options
        load_container = st.container()

        # Option to load existing extraction results
        if st.button("Load Existing Results"):
            # Show a selection interface
            with load_container:
                st.info("Select a saved analysis to load:")

                # List all saved analyses
                saved_analyses = list_saved_raptor_analyses()

                if not saved_analyses:
                    st.warning("No saved analyses found.")
                else:
                    # Create a selectbox with analysis descriptions
                    analysis_options = [analysis["description"] for analysis in saved_analyses]
                    selected_index = st.selectbox(
                        "Select analysis to load",
                        options=range(len(analysis_options)),
                        format_func=lambda i: analysis_options[i]
                    )

                    # Load button
                    if st.button("Load Selected Analysis"):
                        selected_analysis = saved_analyses[selected_index]
                        file_path = selected_analysis["file_path"]

                        # Load the analysis
                        load_result = load_raptor_analysis(file_path)

                        if load_result:
                            individual_results, aggregated_results = load_result

                            # Store in session state
                            st.session_state.extraction_results = individual_results
                            st.session_state.aggregated_results = aggregated_results
                            st.session_state.original_results = aggregated_results.copy()

                            # Initialize feedback for loaded results
                            if st.session_state.aggregated_results:
                                initialize_feedback_for_results(st.session_state.aggregated_results)

                            st.success(f"✅ Loaded analysis: {selected_analysis['description']}")

                            # Refresh the page to show results
                            st.rerun()
                        else:
                            st.error("❌ Failed to load analysis")

    st.divider()
    st.subheader("Saved RAPTOR Analyses")

    # Create columns for better layout
    saved_col1, saved_col2 = st.columns([3, 1])

    with saved_col1:
        # List all saved analyses in a dataframe
        saved_analyses = list_saved_raptor_analyses()

        if not saved_analyses:
            st.info("No saved RAPTOR analyses found.")
        else:
            # Create a dataframe for display
            display_data = []
            for i, analysis in enumerate(saved_analyses):
                display_data.append({
                    "ID": i + 1,
                    "Description": analysis["description"],
                    "Type": "Level Analysis" if analysis["analysis_type"] == "level" else "Cluster Analysis",
                    "Level": analysis["level"] if analysis["level"] is not None else "-",
                    "Cluster": analysis["cluster_id"] if analysis["cluster_id"] else "-",
                    "Date": analysis["timestamp"] if analysis["timestamp"] else "-"
                })

            # Display the dataframe
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True)

    with saved_col2:
        if saved_analyses:
            # Add a selection box to choose which analysis to load
            selected_idx = st.selectbox(
                "Select Analysis",
                options=range(len(saved_analyses)),
                format_func=lambda i: f"ID {i + 1}: {saved_analyses[i]['description'][:20]}..."
            )

            # Add a load button
            if st.button("Load Analysis", key="load_saved_analysis"):
                selected_analysis = saved_analyses[selected_idx]
                file_path = selected_analysis["file_path"]

                # Load the analysis
                load_result = load_raptor_analysis(file_path)

                if load_result:
                    individual_results, aggregated_results = load_result

                    # Store in session state
                    st.session_state.extraction_results = individual_results
                    st.session_state.aggregated_results = aggregated_results
                    st.session_state.original_results = aggregated_results.copy()

                    # Initialize feedback for loaded results
                    if st.session_state.aggregated_results:
                        initialize_feedback_for_results(st.session_state.aggregated_results)

                    st.success(f"✅ Loaded analysis: {selected_analysis['description']}")

                    # Refresh the page to show results
                    st.rerun()
                else:
                    st.error("❌ Failed to load analysis")

    # Display results if available
    if st.session_state.aggregated_results:
        st.divider()
        st.subheader("Extraction Results")

        # Create tabs for different types of extractions
        result_tabs = ["Stakeholders", "Factors", "Pain Points"]

        # Use radio buttons instead of tabs for better visibility
        st.session_state.active_extraction_tab = st.radio(
            "Select data to view:",
            result_tabs,
            index=result_tabs.index(
                st.session_state.active_extraction_tab) if st.session_state.active_extraction_tab in result_tabs else 0
        )

        # Display the selected tab content with feedback interface
        if st.session_state.active_extraction_tab == "Stakeholders":
            # If validation was performed, show before/after summary
            if (st.session_state.validation_results and 
                st.session_state.original_results and 
                "stakeholders" in st.session_state.original_results):
                
                # Display validation results summary
                st.divider()
                st.subheader("Stakeholder Validation Results")
                
                # Calculate validation metrics
                original_count = len(st.session_state.original_results.get("stakeholders", []))
                validated_count = len(st.session_state.aggregated_results.get("stakeholders", []))
                duplicates_merged = st.session_state.validation_results.get("duplicates_merged", 0)
                category_changes = st.session_state.validation_results.get("category_changes", 0)
                
                # Create columns for before/after metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Original Stakeholders", original_count)
                
                with col2:
                    st.metric("Current Stakeholders", validated_count, 
                            delta=validated_count-original_count, 
                            delta_color="inverse")
                
                with col3:
                    st.metric("Duplicates Merged", duplicates_merged)
                
                with col4:
                    st.metric("Categories Corrected", category_changes)
                
                # Show explanation in expander
                with st.expander("About Stakeholder Validation Process", expanded=False):
                    st.markdown("""
                    ### Stakeholder Validation Explained
                    
                    The validation process automatically performs these improvements:
                    
                    1. **Duplicate Detection & Merging**
                       - Identifies potential duplicates based on name similarity
                       - Uses AI to determine if similar names refer to the same entity
                       - Merges duplicates while preserving all information
                    
                    2. **Category Normalization**
                       - Validates stakeholder categories against the standard taxonomy
                       - Corrects inconsistent or invalid classifications
                       - Ensures all stakeholders use approved category labels (Regulator, Supplier, Consumer, Competitor, Partner, Influencer, Internal)
                    
                    All of this happens automatically to improve data quality by resolving inconsistencies and standardizing stakeholder information.
                    """)

            # Create a clean dataframe for display
            if "stakeholders" in st.session_state.aggregated_results:
                stakeholders = st.session_state.aggregated_results["stakeholders"]
                if stakeholders:
                    # Create a simplified DataFrame for display
                    stakeholder_data = []
                    for s in stakeholders:
                        # Format documents info
                        doc_names = []
                        if "documents" in s:
                            doc_names = [d.get("name", "Unknown") for d in s.get("documents", [])]

                        stakeholder_data.append({
                            "Name": s.get("name", ""),
                            "Category": s.get("category", ""),
                            "Role": s.get("role", ""),
                            "Hierarchy Level": s.get("hierarchy_level", ""),
                            "Confidence": s.get("confidence", 0),
                            "Mentions": s.get("mentions", 1),
                            "Documents": ", ".join(doc_names)
                        })

                    # Create and display the DataFrame
                    df = pd.DataFrame(stakeholder_data)
                    st.dataframe(df, use_container_width=True)

                    # Display feedback interface if enabled
                    if st.session_state.show_feedback:
                        st.subheader("Provide Feedback on Stakeholders")
                        st.info(
                            "Please check if you agree with each extraction. If not, uncheck the box and explain why.")

                        # Count statistics
                        agreed_count = 0
                        disagreed_count = 0

                        # Display feedback interface for each stakeholder
                        for i, stakeholder in enumerate(stakeholders):
                            name = stakeholder.get("name", f"Stakeholder {i + 1}")
                            item_id = name

                            with st.expander(f"{i + 1}. {name} ({stakeholder.get('category', 'Unknown')})"):
                                # Show stakeholder details first
                                st.markdown(f"**Role:** {stakeholder.get('role', 'Not specified')}")
                                st.markdown(
                                    f"**Hierarchy Level:** {stakeholder.get('hierarchy_level', 'Not specified')}")
                                st.markdown(f"**Confidence:** {stakeholder.get('confidence', 0)}%")

                                st.divider()

                                # Display feedback interface
                                agrees, feedback = render_feedback_interface(item_id, "stakeholders")

                                # Update statistics
                                if agrees:
                                    agreed_count += 1
                                else:
                                    disagreed_count += 1

                        # Show statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Stakeholders Agreed", agreed_count)
                        with col2:
                            st.metric("Stakeholders Disagreed", disagreed_count)

                    # Provide option to download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Stakeholders CSV",
                        data=csv,
                        file_name="stakeholders_analysis.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No stakeholders found in the extraction results.")
            else:
                st.info("No stakeholder data available in the extraction results.")

        elif st.session_state.active_extraction_tab == "Factors":
            # Create a clean dataframe for display
            if "factors" in st.session_state.aggregated_results:
                factors = st.session_state.aggregated_results["factors"]
                if factors:
                    # Create a simplified DataFrame for display
                    factor_data = []
                    for f in factors:
                        # Get document info
                        doc_name = "Unknown"
                        if "source_document" in f:
                            doc_name = f.get("source_document", {}).get("name", "Unknown")

                        factor_data.append({
                            "Label": f.get("Label", ""),
                            "Type": f.get("Type", ""),
                            "Tags": f.get("Tags", ""),
                            "Description": f.get("Description", ""),
                            "Source Document": doc_name
                        })

                    # Create and display the DataFrame
                    df = pd.DataFrame(factor_data)
                    st.dataframe(df, use_container_width=True)

                    # Display feedback interface if enabled
                    if st.session_state.show_feedback:
                        st.subheader("Provide Feedback on Factors")
                        st.info(
                            "Please check if you agree with each extraction. If not, uncheck the box and explain why.")

                        # Count statistics
                        agreed_count = 0
                        disagreed_count = 0

                        # Display feedback interface for each factor
                        for i, factor in enumerate(factors):
                            label = factor.get("Label", f"Factor {i + 1}")
                            item_id = label

                            with st.expander(f"{i + 1}. {label} ({factor.get('Type', 'Unknown')})"):
                                # Show factor details first
                                st.markdown(f"**Tags:** {factor.get('Tags', 'None')}")
                                st.markdown(f"**Description:** {factor.get('Description', 'Not specified')}")

                                if "source_document" in factor:
                                    doc_name = factor.get("source_document", {}).get("name", "Unknown")
                                    st.markdown(f"**Source:** {doc_name}")

                                st.divider()

                                # Display feedback interface
                                agrees, feedback = render_feedback_interface(item_id, "factors")

                                # Update statistics
                                if agrees:
                                    agreed_count += 1
                                else:
                                    disagreed_count += 1

                        # Show statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Factors Agreed", agreed_count)
                        with col2:
                            st.metric("Factors Disagreed", disagreed_count)

                    # Provide option to download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Factors CSV",
                        data=csv,
                        file_name="factors_analysis.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No factors found in the extraction results.")
            else:
                st.info("No factors data available in the extraction results.")

        elif st.session_state.active_extraction_tab == "Pain Points":
            # Create a clean dataframe for display
            if "pain_points" in st.session_state.aggregated_results:
                pain_points = st.session_state.aggregated_results["pain_points"]
                if pain_points:
                    # Create a simplified DataFrame for display
                    pain_point_data = []
                    for p in pain_points:
                        # Get document info
                        doc_name = "Unknown"
                        if "source_document" in p:
                            doc_name = p.get("source_document", {}).get("name", "Unknown")

                        pain_point_data.append({
                            "ID": p.get("id", ""),
                            "Category": p.get("category", ""),
                            "Description": p.get("description", ""),
                            "Confidence": p.get("confidence", 0),
                            "Hierarchy Level": p.get("hierarchy_level", ""),
                            "Source Document": doc_name
                        })

                    # Create and display the DataFrame
                    df = pd.DataFrame(pain_point_data)
                    st.dataframe(df, use_container_width=True)

                    # Display feedback interface if enabled
                    if st.session_state.show_feedback:
                        st.subheader("Provide Feedback on Pain Points")
                        st.info(
                            "Please check if you agree with each extraction. If not, uncheck the box and explain why.")

                        # Count statistics
                        agreed_count = 0
                        disagreed_count = 0

                        # Display feedback interface for each pain point
                        for i, pain_point in enumerate(pain_points):
                            item_id = pain_point.get("id", f"pain_point_{i}")
                            category = pain_point.get("category", "Unknown")

                            with st.expander(f"{i + 1}. {item_id} - {category}"):
                                # Show pain point details first
                                st.markdown(f"**Description:** {pain_point.get('description', 'Not specified')}")
                                st.markdown(f"**Confidence:** {pain_point.get('confidence', 0)}%")
                                st.markdown(
                                    f"**Hierarchy Level:** {pain_point.get('hierarchy_level', 'Not specified')}")

                                if "source_document" in pain_point:
                                    doc_name = pain_point.get("source_document", {}).get("name", "Unknown")
                                    st.markdown(f"**Source:** {doc_name}")

                                st.divider()

                                # Display feedback interface
                                agrees, feedback = render_feedback_interface(item_id, "pain_points")

                                # Update statistics
                                if agrees:
                                    agreed_count += 1
                                else:
                                    disagreed_count += 1

                        # Show statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Pain Points Agreed", agreed_count)
                        with col2:
                            st.metric("Pain Points Disagreed", disagreed_count)

                    # Provide option to download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Pain Points CSV",
                        data=csv,
                        file_name="pain_points_analysis.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No pain points found in the extraction results.")
            else:
                st.info("No pain points data available in the extraction results.")
        
        # Feedback summary and export - Enhanced with learning
        if st.session_state.show_feedback and hasattr(st.session_state, "user_feedback"):
            st.divider()
            st.subheader("User Feedback Summary & Learning")

            # Collect all disagreements (keep your existing code for this part)
            disagreements = []

            # Check stakeholders
            for item_id, feedback in st.session_state.user_feedback["stakeholders"].items():
                if not feedback.get("agrees", True):
                    disagreements.append({
                        "Type": "Stakeholder",
                        "ID": item_id,
                        "Feedback": feedback.get("feedback", "")
                    })

            # Check factors
            for item_id, feedback in st.session_state.user_feedback["factors"].items():
                if not feedback.get("agrees", True):
                    disagreements.append({
                        "Type": "Factor",
                        "ID": item_id,
                        "Feedback": feedback.get("feedback", "")
                    })

            # Check pain points
            for item_id, feedback in st.session_state.user_feedback["pain_points"].items():
                if not feedback.get("agrees", True):
                    disagreements.append({
                        "Type": "Pain Point",
                        "ID": item_id,
                        "Feedback": feedback.get("feedback", "")
                    })

            # Display disagreements with enhanced learning options
            if disagreements:
                st.markdown(f"**Total disagreements:** {len(disagreements)}")

                # Show disagreements table
                st.dataframe(pd.DataFrame(disagreements), use_container_width=True)

                # Enhanced feedback storage with learning
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Save Feedback for Learning"):
                        if not st.session_state.feedback_system_initialized:
                            st.warning("Please initialize the feedback system first")
                        else:
                            try:
                                # Process and store feedback
                                success = process_streamlit_feedback(
                                    st.session_state.feedback_agent,
                                    st.session_state.user_feedback,
                                    st.session_state.aggregated_results
                                )
                                
                                if success:
                                    st.success("✅ Feedback stored for learning!")
                                    
                                    # Show updated stats
                                    stats = st.session_state.feedback_agent.feedback_storage.get_feedback_stats()
                                    st.info(f"System now has {stats['total_feedback']} total feedback records")
                                else:
                                    st.error("❌ Failed to store feedback")
                            except Exception as e:
                                st.error(f"❌ Error storing feedback: {e}")
                
                with col2:
                    if st.button("Apply Learned Improvements"):
                        if not st.session_state.feedback_system_initialized:
                            st.warning("Please initialize the feedback system first")
                        else:
                            try:
                                with st.spinner("Applying learned feedback..."):
                                    # Reprocess using feedback
                                    improved_results = async_to_sync(reprocess_with_feedback(
                                        st.session_state.feedback_agent,
                                        st.session_state.aggregated_results
                                    ))
                                    
                                    # Store improved results
                                    st.session_state.aggregated_results = improved_results
                                    
                                    st.success("✅ Results improved with learned feedback!")
                                    st.info("Classifications have been updated based on previous feedback")
                                    
                                    # Refresh the display
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error applying improvements: {e}")
                
                with col3:
                    # Export feedback button (keep your existing code)
                    if st.button("Export Feedback JSON"):
                        feedback_file = save_user_feedback()
                        if feedback_file:
                            st.success(f"✅ Feedback exported to {feedback_file}")

                            # Offer to download the feedback file
                            with open(feedback_file, "r") as f:
                                feedback_json = f.read()

                            st.download_button(
                                label="Download Feedback JSON",
                                data=feedback_json,
                                file_name=f"user_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        else:
                            st.error("❌ Failed to export feedback")
            else:
                st.info("No disagreements recorded. Use the checkboxes in each section to provide feedback.")
        
        # Display download options for full JSON results - RESTORED SECTION
        st.divider()
        st.subheader("Download Full Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Option to download full workspace analysis
            json_str = json.dumps(st.session_state.aggregated_results, indent=2)
            st.download_button(
                label="Download Full Analysis JSON",
                data=json_str,
                file_name="analysis_results.json",
                mime="application/json",
            )

        with col2:
            # Option to download just stakeholders
            if "stakeholders" in st.session_state.aggregated_results:
                stakeholders_data = {
                    "total_documents": st.session_state.aggregated_results.get("total_documents", 0),
                    "stakeholders": st.session_state.aggregated_results.get("stakeholders", [])
                }
                json_str = json.dumps(stakeholders_data, indent=2)
                st.download_button(
                    label="Download Stakeholders JSON",
                    data=json_str,
                    file_name="stakeholders_only.json",
                    mime="application/json",
                )

        with col3:
            # Option to download individual document extractions
            if st.session_state.extraction_results:
                first_extraction = next(iter(st.session_state.extraction_results.values()))
                json_str = json.dumps(first_extraction, indent=2)
                st.download_button(
                    label="Download Document Extraction Sample",
                    data=json_str,
                    file_name="document_extraction_sample.json",
                    mime="application/json",
                )

        # Visualization section
        if "ecosystem_map" in st.session_state.aggregated_results:
            st.divider()
            st.subheader("Ecosystem Visualization")

            # Basic visualization of the ecosystem map
            eco_map = st.session_state.aggregated_results["ecosystem_map"]["ecosystemMap"]

            # Display hierarchy stats
            col1, col2, col3 = st.columns(3)

            with col1:
                macro_count = len(eco_map["levels"]["macro"]["nodes"])
                st.metric("Macro Level Stakeholders", macro_count)

            with col2:
                meso_count = len(eco_map["levels"]["meso"]["nodes"])
                st.metric("Meso Level Stakeholders", meso_count)

            with col3:
                micro_count = len(eco_map["levels"]["micro"]["nodes"])
                st.metric("Micro Level Stakeholders", micro_count)

            # Display relationships visualization if available
            if "relationships" in eco_map and eco_map["relationships"]:
                st.subheader("Stakeholder Relationships")

                # Convert relationships to a format suitable for a network graph
                relationships = eco_map["relationships"]

                # Create nodes and edges for network visualization
                nodes = []
                node_ids = set()

                # Collect all nodes from relationships
                for rel in relationships:
                    source = rel.get("source", "")
                    target = rel.get("target", "")

                    if source and source not in node_ids:
                        nodes.append({
                            "id": source,
                            "level": rel.get("source_level", "unknown")
                        })
                        node_ids.add(source)

                    if target and target not in node_ids:
                        nodes.append({
                            "id": target,
                            "level": rel.get("target_level", "unknown")
                        })
                        node_ids.add(target)

                # Create a simplified relationship view
                st.info(f"Found {len(relationships)} relationships between stakeholders")

                # Show relationship table
                rel_data = []
                for rel in relationships:
                    rel_data.append({
                        "From": rel.get("source", ""),
                        "From Level": rel.get("source_level", "").capitalize(),
                        "To": rel.get("target", ""),
                        "To Level": rel.get("target_level", "").capitalize(),
                        "Type": rel.get("type", ""),
                        "Strength": rel.get("strength", 0)
                    })

                if rel_data:
                    st.dataframe(pd.DataFrame(rel_data), use_container_width=True)

# Tab 6: Feedback Management (keeping existing code)
with tab6:
    st.header("Feedback Management & Learning")
    
    if not st.session_state.feedback_system_initialized:
        st.warning("Please initialize the feedback system in the sidebar first.")
        
        if st.button("Initialize Feedback System Here"):
            try:
                st.session_state.feedback_agent = initialize_feedback_system()
                st.session_state.feedback_system_initialized = True
                st.success("✅ Feedback system initialized!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to initialize: {e}")
    else:
        # Performance Overview
        st.subheader("📊 Learning Performance Overview")
        
        if st.session_state.feedback_agent:
            try:
                stats = st.session_state.feedback_agent.feedback_storage.get_feedback_stats()
                
                if stats['total_feedback'] > 0:
                    # Create metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Feedback", stats['total_feedback'])
                    with col2:
                        st.metric("Accuracy Rate", f"{stats['accuracy_rate']:.1%}")
                    with col3:
                        st.metric("Positive Feedback", stats['positive_feedback'])
                    with col4:
                        st.metric("Needs Improvement", stats['negative_feedback'])
                    
                    # Performance by type
                    if stats.get('by_type'):
                        st.subheader("Performance by Type")
                        
                        type_data = []
                        for ext_type, type_stats in stats['by_type'].items():
                            accuracy = type_stats['positive'] / type_stats['total'] if type_stats['total'] > 0 else 0
                            type_data.append({
                                "Type": ext_type.capitalize(),
                                "Total Feedback": type_stats['total'],
                                "Accuracy": f"{accuracy:.1%}",
                                "Needs Work": type_stats['negative']
                            })
                        
                        st.dataframe(pd.DataFrame(type_data), use_container_width=True)
                else:
                    st.info("No feedback data available yet. Start by providing feedback on extraction results.")
            
            except Exception as e:
                st.error(f"Error loading performance data: {e}")
        
        # [Rest of feedback management tab remains the same]...

# Helper function for async operations
def async_to_sync(async_func):
    """Helper function to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(async_func)

# Footer
st.divider()
st.markdown("*Raptor FAISS Retriever & Analysis Tool with DOCX Directory Processing*")