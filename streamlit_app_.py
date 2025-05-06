import streamlit as st
import asyncio
import os
import json
import pandas as pd
from dotenv import load_dotenv
import time

from big_raptor.raptor_visualization import display_raptor_visualization, generate_hierarchy_data
# Import our services
from services.supabase_service import SupabaseService
from services.raptor_service import RaptorService
from services.document_service import prepare_documents, convert_supabase_to_langchain

# Import raptor components
from big_raptor.base import QueryModes
from utils.visualization import create_cluster_visualizations

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Raptor FAISS Retriever",
    page_icon="🦖",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
st.title("🦖 Raptor FAISS Retriever")
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
        """)

# Main content area
tab1, tab2, tab3 = st.tabs([
    "Document Selection",
    "Raptor Processing",
    "Raptor Visualization"  # New tab
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
                            process_status = asyncio.run(
                                st.session_state.raptor_service.index_documents(
                                    langchain_docs,
                                    namespace=namespace,
                                    fresh_start=fresh_start
                                )
                            )

                            # Update indexing status
                            st.session_state.indexed_status = f"Indexed {len(langchain_docs)} documents"
                            st.success(f"Successfully processed {len(langchain_docs)} documents with Raptor")
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

with tab3:
    if st.session_state.indexed_status == "Not indexed":
        st.warning("Please index documents with Raptor in the Processing tab first.")
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

st.divider()
st.markdown("*Raptor FAISS Retriever Testing UI*")