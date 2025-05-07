import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import raptor components
from big_raptor.base import QueryModes
from big_raptor.raptor_visualization import display_raptor_visualization, generate_hierarchy_data
# Import the new StakeholderAnalyzer
from core.raptor_stakeholder_analyzer import RaptorStakeholderAnalyzer, analyze_raptor_level, analyze_raptor_cluster
from services.document_service import convert_supabase_to_langchain
from services.raptor_service import RaptorService
# Import our services
from services.supabase_service import SupabaseService

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
    st.session_state.extraction_model = "openai/gpt-4o"

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

if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = True


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
        ["openai/gpt-4o", "openai/gpt-4", "anthropic/claude-3-opus", "anthropic/claude-3-sonnet"],
        index=0
    )

    # Feedback configuration
    st.subheader("Feedback Options")

    # Toggle for showing feedback interface
    st.session_state.show_feedback = st.checkbox(
        "Enable User Feedback Interface",
        value=st.session_state.show_feedback,
        help="Show checkboxes and comment fields for providing feedback on extraction results"
    )

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
        """)

    st.subheader("RAPTOR Analysis Options")

    # Extend extraction mode to include RAPTOR options
    st.session_state.extraction_mode = st.sidebar.radio(
        "Extraction Mode",
        ["Selected Documents", "Direct Text Input", "RAPTOR Clusters"],
        index=0,
        help="Choose source for text analysis"
    )

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
tab1, tab2, tab3, tab4 = st.tabs([
    "Document Selection",
    "Raptor Processing",
    "Raptor Visualization",
    "Extraction Analysis"
])

# Tab 1: Document Selection
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

# Tab 2: Raptor Processing
with tab2:
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

# Tab 3: Raptor Visualization
with tab3:
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

# Tab 4: Extraction Analysis
with tab4:
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

                    # Initialize feedback for loaded results
                    if st.session_state.aggregated_results:
                        initialize_feedback_for_results(st.session_state.aggregated_results)

                    st.success(f"✅ Loaded analysis: {selected_analysis['description']}")

                    # Refresh the page to show results
                    st.experimental_rerun()
                else:
                    st.error("❌ Failed to load analysis")
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
                                    output_dir=output_dir,
                                    namespace=st.session_state.raptor_service.namespace,
                                    verbose=True
                                ))

                                if result.get('error'):
                                    st.error(f"Error analyzing cluster: {result.get('error')}")
                                else:
                                    # Store results in session state
                                    st.session_state.extraction_results = {st.session_state.selected_cluster: result}
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
                                    output_dir=output_dir,
                                    namespace=st.session_state.raptor_service.namespace,
                                    verbose=True,
                                    aggregate=True
                                ))

                                if results.get('error'):
                                    st.error(f"Error analyzing level {level}: {results.get('error')}")
                                else:
                                    # Store results in session state
                                    st.session_state.extraction_results = results.get('individual_results', {})
                                    st.session_state.aggregated_results = results.get('aggregated_results', {})

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
            else:
                # Original code for document/text-based extraction
                success = run_extraction_analysis()
                if success:
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

                            # Initialize feedback for loaded results
                            if st.session_state.aggregated_results:
                                initialize_feedback_for_results(st.session_state.aggregated_results)

                            st.success(f"✅ Loaded analysis: {selected_analysis['description']}")

                            # Refresh the page to show results
                            st.experimental_rerun()
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

        # Feedback summary and export
        if st.session_state.show_feedback and hasattr(st.session_state, "user_feedback"):
            st.divider()
            st.subheader("User Feedback Summary")

            # Collect all disagreements
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

            # Display disagreements
            if disagreements:
                st.markdown(f"**Total disagreements:** {len(disagreements)}")

                # Show disagreements table
                st.dataframe(pd.DataFrame(disagreements), use_container_width=True)

                # Export feedback button
                if st.button("Save User Feedback"):
                    feedback_file = save_user_feedback()
                    if feedback_file:
                        st.success(f"✅ Feedback saved to {feedback_file}")

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
                        st.error("❌ Failed to save feedback")
            else:
                st.info("No disagreements recorded. Use the checkboxes in each section to provide feedback.")

        # Display download options for full JSON results
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
    else:
        # If no results, show example of the expected output
        st.divider()
        st.subheader("Example Output")

        # Show tabs for different types
        example_tabs = st.tabs(["Stakeholders Example", "Factors Example", "Pain Points Example"])

        with example_tabs[0]:
            st.markdown("### Stakeholders Table Example")

            # Create example stakeholder data
            example_stakeholders = [
                {
                    "Name": "HSBC",
                    "Category": "Supplier",
                    "Role": "Provides banking services and products",
                    "Hierarchy Level": "Meso",
                    "Confidence": 95,
                    "Mentions": 3,
                    "Documents": "2019_HSBC age-friendly banking.pdf, 2021_HSBC_productive-ageing-in-hong-kong.pdf"
                },
                {
                    "Name": "Census and Statistics Department",
                    "Category": "Regulator",
                    "Role": "Provides population projections",
                    "Hierarchy Level": "Macro",
                    "Confidence": 90,
                    "Mentions": 1,
                    "Documents": "2019_HSBC age-friendly banking.pdf"
                }
            ]

            # Display example
            st.dataframe(pd.DataFrame(example_stakeholders), use_container_width=True)

            # Show feedback UI example
            st.subheader("Feedback Interface Example")

            with st.expander("Example: HSBC (Supplier)"):
                st.markdown("**Role:** Provides banking services and products")
                st.markdown("**Hierarchy Level:** Meso")
                st.markdown("**Confidence:** 95%")

                st.divider()

                col1, col2 = st.columns([1, 5])
                with col1:
                    st.checkbox("Agree", value=True, key="example_checkbox_1", disabled=True)


        with example_tabs[1]:
            st.markdown("### Factors Table Example")

            # Create example factor data
            example_factors = [
                {
                    "Label": "Aging Population in Hong Kong",
                    "Type": "Exogenous Driver",
                    "Tags": "Demographic Shift",
                    "Description": "The increasing number of elderly individuals in Hong Kong",
                    "Source Document": "2019_HSBC age-friendly banking.pdf"
                },
                {
                    "Label": "Financial Crime",
                    "Type": "Direct Driver",
                    "Tags": "Fraud Protection",
                    "Description": "The need to protect financial accounts from scams and fraud",
                    "Source Document": "2019_HSBC age-friendly banking.pdf"
                }
            ]

            # Display example
            st.dataframe(pd.DataFrame(example_factors), use_container_width=True)

            # Show feedback UI example
            with st.expander("Example: Financial Crime (Direct Driver)"):
                st.markdown("**Tags:** Fraud Protection")
                st.markdown("**Description:** The need to protect financial accounts from scams and fraud")
                st.markdown("**Source:** 2019_HSBC age-friendly banking.pdf")

                st.divider()

                col1, col2 = st.columns([1, 5])
                with col1:
                    st.checkbox("Agree", value=False, key="example_checkbox_2", disabled=True)
                with col2:
                    st.text_area("Please explain why you disagree:",
                                 value="This should be classified as an Outcome, not a Direct Driver, as it's a result of other factors.",
                                 key="example_feedback_2",
                                 height=100,
                                 disabled=True)

        with example_tabs[2]:
            st.markdown("### Pain Points Table Example")

            # Create example pain point data
            example_pain_points = [
                {
                    "ID": "PP-001",
                    "Category": "Security Concerns",
                    "Description": "Customers are vulnerable to scams and fraud",
                    "Confidence": 95,
                    "Hierarchy Level": "Micro",
                    "Source Document": "2019_HSBC age-friendly banking.pdf"
                },
                {
                    "ID": "PP-002",
                    "Category": "Digital Banking Adoption",
                    "Description": "Older customers may find it challenging to adjust to digital banking",
                    "Confidence": 92,
                    "Hierarchy Level": "Micro",
                    "Source Document": "2019_HSBC age-friendly banking.pdf"
                }
            ]

            # Display example
            st.dataframe(pd.DataFrame(example_pain_points), use_container_width=True)

        # Explain the process
        st.info("""
        **How the extraction and feedback process works:**

        1. Choose your extraction mode in the sidebar: "Selected Documents" or "Direct Text Input"
        2. If using documents, select them in the Document Selection tab
        3. If using direct text input, add your text content in the form above
        4. Click 'Run Extraction Analysis' to process the text
        5. The system will extract stakeholders, factors, and pain points from your text
        6. Review the extractions and provide feedback:
           - Check the box if you agree with the extraction
           - Uncheck and provide comments if you disagree
        7. Save your feedback to help improve the extraction process
        """)

st.divider()
st.markdown("*Raptor FAISS Retriever & Analysis Tool*")