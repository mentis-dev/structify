# big_raptor/raptor_visualization.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from typing import Dict, Any, List, Optional


def generate_hierarchy_data(retriever, namespace: str = ""):
    """Generate structured data representing the document hierarchy."""
    hierarchy_data = {
        "levels": {},
        "total_documents": 0,
        "level_distribution": {},
        "metadata": {
            "tree_depth": retriever.tree_depth,
            "clustering_threshold": getattr(retriever, "clustering_threshold", 0.1),
            "max_length_in_cluster": getattr(retriever, "max_length_in_cluster", 10000),
            "mode": retriever.mode
        },
        "parent_child_relationships": []  # Track parent-child links
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
        sample_size = min(20, level_distribution[level])
        try:
            docs = retriever._get_documents_by_metadata("level", level, namespace=namespace)
            sample_docs = docs[:sample_size] if docs else []

            # Extract relevant info
            level_docs = []
            for doc in sample_docs:
                doc_id = doc.id or doc.metadata.get("document_id", "unknown")

                doc_info = {
                    "id": doc_id,
                    "content_preview": doc.page_content[:200] + "..." if len(
                        doc.page_content) > 200 else doc.page_content,
                    "content": doc.page_content,
                    "metadata": {k: v for k, v in doc.metadata.items()
                                 if k not in ["embedding"] and not isinstance(v, (list, np.ndarray))}
                }

                # For level > 0, get child count
                if level > 0 and "child_count" in doc.metadata:
                    doc_info["child_count"] = doc.metadata["child_count"]

                level_docs.append(doc_info)

                # Collect parent-child relationships if parent_id exists
                if "parent_id" in doc.metadata:
                    hierarchy_data["parent_child_relationships"].append({
                        "parent_id": doc.metadata["parent_id"],
                        "child_id": doc_id,
                        "parent_level": level + 1,  # Parent is one level up
                        "child_level": level
                    })

            hierarchy_data["levels"][level] = {
                "document_count": level_distribution[level],
                "sample_documents": level_docs
            }
        except Exception as e:
            st.error(f"Error getting documents for level {level}: {str(e)}")
            hierarchy_data["levels"][level] = {
                "document_count": level_distribution[level],
                "sample_documents": [],
                "error": str(e)
            }

    return hierarchy_data


def create_tree_visualization(hierarchy_data):
    """Create a hierarchical tree visualization of the document structure."""
    # Extract levels
    levels = sorted([int(level) for level in hierarchy_data["levels"].keys()])
    max_level = max(levels) if levels else 0

    # Initialize figure
    fig = go.Figure()

    # Node tracking
    nodes = []
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_sizes = []
    node_colors = []
    node_ids = {}  # Map document IDs to node indices

    # Edge tracking
    edges_x = []
    edges_y = []
    edge_colors = []

    # Color map for levels
    colors = px.colors.qualitative.Set1

    # Layout parameters
    level_height = 1.0
    node_spacing = 1.5

    # Add level nodes first
    for level in reversed(levels):  # Start from highest level
        level_data = hierarchy_data["levels"][level]
        doc_count = level_data["document_count"]

        # Position this level's node
        y_pos = (max_level - level) * level_height * 3

        # Add the level node
        node_ids[f"level_{level}"] = len(nodes)
        nodes.append(f"level_{level}")
        node_x.append(0)  # Center
        node_y.append(y_pos)
        node_text.append(f"Level {level}")
        node_info.append(f"Level {level}: {doc_count} documents")
        node_sizes.append(30 + min(50, doc_count / 5))  # Scale with doc count, but cap it
        node_colors.append(colors[level % len(colors)])

        # Add sample documents for this level
        sample_docs = level_data.get("sample_documents", [])[:8]  # Limit samples
        if sample_docs:
            num_docs = len(sample_docs)
            width = num_docs * node_spacing

            for i, doc in enumerate(sample_docs):
                # Position docs in a row
                doc_x = ((i / (num_docs - 1)) - 0.5) * width if num_docs > 1 else 0
                doc_y = y_pos + 1  # Slightly below level node

                # Add doc node
                doc_id = doc["id"]
                node_ids[doc_id] = len(nodes)
                nodes.append(doc_id)
                node_x.append(doc_x)
                node_y.append(doc_y)

                # Prepare display text
                preview = doc.get("content_preview", "")[:50]
                if len(preview) > 47:
                    preview = preview[:47] + "..."

                node_text.append(f"Doc {i + 1}")

                # Prepare hover info
                hover_info = f"ID: {doc_id[-12:] if len(doc_id) > 12 else doc_id}"
                if "child_count" in doc:
                    hover_info += f"<br>Children: {doc['child_count']}"
                hover_info += f"<br>{preview}"

                node_info.append(hover_info)
                node_sizes.append(15)  # Smaller than level nodes
                node_colors.append(colors[level % len(colors)])

                # Add edge from level to doc
                edges_x.extend([node_x[node_ids[f"level_{level}"]], doc_x, None])
                edges_y.extend([node_y[node_ids[f"level_{level}"]], doc_y, None])
                edge_colors.append("rgba(150,150,150,0.5)")

    # Add level connections
    for i in range(len(levels) - 1):
        level1 = levels[i]
        level2 = levels[i + 1]

        # Connect level nodes
        edges_x.extend([node_x[node_ids[f"level_{level1}"]], node_x[node_ids[f"level_{level2}"]], None])
        edges_y.extend([node_y[node_ids[f"level_{level1}"]], node_y[node_ids[f"level_{level2}"]], None])
        edge_colors.append("rgba(100,100,100,0.8)")

    # Add parent-child relationships
    for relation in hierarchy_data["parent_child_relationships"]:
        parent_id = relation["parent_id"]
        child_id = relation["child_id"]

        # Only add if both nodes are in our visualization
        if parent_id in node_ids and child_id in node_ids:
            parent_idx = node_ids[parent_id]
            child_idx = node_ids[child_id]

            edges_x.extend([node_x[parent_idx], node_x[child_idx], None])
            edges_y.extend([node_y[parent_idx], node_y[child_idx], None])
            edge_colors.append("rgba(255,165,0,0.6)")  # Orange for parent-child

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='white')
        ),
        text=node_text,
        textposition="top center",
        hovertext=node_info,
        hoverinfo='text',
        name='Nodes'
    ))

    # Add edges
    if edges_x:
        fig.add_trace(go.Scatter(
            x=edges_x,
            y=edges_y,
            mode='lines',
            line=dict(color='rgba(150,150,150,0.5)', width=1),
            hoverinfo='none',
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title="Raptor Document Hierarchy Tree",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )

    return fig


def create_level_pie_chart(hierarchy_data):
    """Create a pie chart showing document distribution by level."""
    labels = []
    values = []

    for level, count in hierarchy_data["level_distribution"].items():
        if isinstance(level, int):
            labels.append(f"Level {level}")
            values.append(count)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        textinfo='label+percent',
        marker=dict(
            colors=px.colors.qualitative.Set1[:len(labels)]
        )
    )])

    fig.update_layout(
        title="Document Distribution by Level",
        height=400
    )

    return fig


def create_sunburst_chart(hierarchy_data):
    """Create a sunburst chart of the document hierarchy."""

    # Add this debugging statement
    st.write("Starting sunburst chart creation...")

    try:
        # Check if hierarchy data is valid
        if not hierarchy_data:
            st.warning("Hierarchy data is empty or None")
            return None

        if "levels" not in hierarchy_data:
            st.warning("No 'levels' key in hierarchy data")
            return None

        # Debug the levels data
        st.write(f"Found {len(hierarchy_data['levels'])} levels in hierarchy data")
        st.write(f"Level keys: {list(hierarchy_data['levels'].keys())}")

        # Prepare data
        labels = ["Root"]
        parents = [""]
        values = [1]  # Root value
        colors = []  # Custom colors for nodes

        # Color scale
        level_colors = px.colors.qualitative.Set1

        # Add level nodes
        for level in sorted(hierarchy_data["levels"].keys(), key=int):
            level_int = int(level)
            doc_count = hierarchy_data["levels"][level]["document_count"]

            # Debug each level's data
            st.write(f"Processing Level {level_int}: {doc_count} documents")

            if doc_count > 0:  # Only add levels with documents
                level_name = f"Level {level_int}"
                labels.append(level_name)
                parents.append("Root")
                values.append(doc_count)
                colors.append(level_colors[level_int % len(level_colors)])

                # Add sample documents for top levels only (to avoid clutter)
                if level_int >= max(int(l) for l in hierarchy_data["levels"].keys()) - 1:
                    sample_docs = hierarchy_data["levels"][level].get("sample_documents", [])
                    st.write(f"Adding {min(5, len(sample_docs))} sample docs for Level {level_int}")

                    for i, doc in enumerate(sample_docs[:5]):  # Limit to 5 samples per level
                        doc_id = f"Doc {i + 1} (L{level_int})"
                        labels.append(doc_id)
                        parents.append(level_name)
                        values.append(1)  # Each document has weight 1
                        colors.append(level_colors[level_int % len(level_colors)])

        # If we don't have enough data, return None
        if len(labels) <= 1:
            st.warning("Not enough label data for sunburst chart")
            return None

        st.write(f"Created {len(labels)} labels for sunburst chart")

        # Ensure colors has the right length
        if len(colors) < len(labels):
            st.write(f"Adding {len(labels) - len(colors)} missing colors")
            colors = colors + [level_colors[0]] * (len(labels) - len(colors))

        # Display data arrays for debugging
        st.write("Data prepared for sunburst chart:")
        st.write({
            "Labels": labels[:10] + ["..."] if len(labels) > 10 else labels,
            "Parents": parents[:10] + ["..."] if len(parents) > 10 else parents,
            "Values": values[:10] + ["..."] if len(values) > 10 else values,
            "Colors count": len(colors)
        })

        # Create sunburst chart
        st.write("Creating Plotly figure...")
        fig = go.Figure()

        fig.add_trace(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors
            ),
            hoverinfo="label+value",
            maxdepth=3  # Limit depth to avoid too much detail
        ))

        fig.update_layout(
            title="Document Hierarchy Sunburst",
            height=500,
            margin=dict(t=30, l=0, r=0, b=0)
        )

        st.write("Sunburst chart created successfully")
        return fig

    except Exception as e:
        # Log the error and return None
        st.error(f"Error creating sunburst chart: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None



def create_abstraction_flow(hierarchy_data):
    """Create visualization of the abstraction process from level 0 to top level."""
    # Build data for each level
    level_stats = {}
    for level, data in hierarchy_data["levels"].items():
        level_int = int(level)
        docs = data.get("sample_documents", [])

        if docs:
            # Get stats about doc length at this level
            lengths = [len(doc.get("content_preview", "")) for doc in docs]
            avg_length = sum(lengths) / len(lengths) if lengths else 0

            level_stats[level_int] = {
                "doc_count": data["document_count"],
                "avg_length": avg_length,
                "example": docs[0].get("content_preview", "No content") if docs else ""
            }

    # Create bar chart of average document length by level
    level_df = pd.DataFrame([
        {"Level": level, "Average Document Length": stats["avg_length"],
         "Document Count": stats["doc_count"]}
        for level, stats in level_stats.items()
    ])

    # Sort by level
    if not level_df.empty:
        level_df = level_df.sort_values("Level")

        # Create bar chart
        fig = px.bar(
            level_df,
            x="Level",
            y="Average Document Length",
            color="Level",
            hover_data=["Document Count"],
            title="Text Abstraction: Length by Level",
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        fig.update_layout(height=400)
        return fig, level_stats

    return None, level_stats


def treemap_viz(hierarchy_data):
    """Try a treemap visualization instead of sunburst"""

    st.subheader("Treemap Visualization")

    # Prepare data
    levels = hierarchy_data.get("levels", {})

    # Create a DataFrame for the treemap
    data = []

    for key in sorted([int(k) for k in levels.keys()]):
        level_name = f"Level {key}"
        doc_count = levels[key].get("document_count", 0)

        if doc_count > 0:
            data.append({
                "level": level_name,
                "count": doc_count
            })

    df = pd.DataFrame(data)

    # Create treemap
    try:
        fig = px.treemap(
            df,
            path=["level"],
            values="count",
            title="Document Hierarchy Treemap"
        )

        fig.update_layout(height=600)

        return fig
    except Exception as e:
        st.error(f"Error creating treemap: {e}")
        return None



def display_raptor_visualization(raptor_service, namespace="default"):
    """Main function to display the Raptor visualization dashboard in Streamlit."""

    st.header("Raptor Document Hierarchy Visualization")

    with st.spinner("Generating hierarchy visualization..."):
        try:
            # Check if hierarchy data is already in session_state
            if "hierarchy_data" in st.session_state:
                hierarchy_data = st.session_state.hierarchy_data
            else:
                # Generate hierarchy data
                hierarchy_data = generate_hierarchy_data(raptor_service.retriever, namespace)
                st.session_state.hierarchy_data = hierarchy_data

            # Display summary stats
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Total Documents",
                    value=hierarchy_data["total_documents"]
                )

            with col2:
                st.metric(
                    label="Tree Depth",
                    value=hierarchy_data["metadata"]["tree_depth"]
                )

            with col3:
                # Count total levels that have documents
                active_levels = sum(1 for level, count in hierarchy_data["level_distribution"].items()
                                    if isinstance(level, int) and count > 0)
                st.metric(
                    label="Active Levels",
                    value=active_levels
                )

            # Create tabs for different visualizations
            viz_tabs = st.tabs([
                "Tree Structure",
                "Document Distribution",
                "Recursive Abstraction",
                "Level Details"
            ])

            # Tree Structure Tab
            with viz_tabs[0]:
                st.subheader("Document Hierarchy Tree")
                tree_fig = create_tree_visualization(hierarchy_data)
                st.plotly_chart(tree_fig, use_container_width=True, key="tree_structure_main")

                st.info("""
                **Understanding the Tree Structure**

                The Raptor algorithm builds a hierarchical tree where:

                - **Levels**: Each horizontal layer represents a level in the hierarchy
                - **Documents**: Smaller nodes represent individual documents
                - **Connections**: Lines show relationships between levels and documents

                The tree is built from the bottom up, with documents at Level 0 being clustered and summarized
                to create documents at Level 1, and so on up to the root level.
                """)

            # Document Distribution Tab
            with viz_tabs[1]:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Document Count by Level")
                    # Create level distribution chart
                    level_data = {}
                    for level, count in hierarchy_data["level_distribution"].items():
                        if isinstance(level, int):
                            level_data[f"Level {level}"] = count

                    if level_data:
                        # Create a container specifically for the bar chart
                        bar_chart_container = st.container()
                        with bar_chart_container:
                            st.bar_chart(level_data)
                    else:
                        st.warning("No level data available")

                with col2:
                    st.subheader("Proportional Distribution")
                    pie_fig = create_level_pie_chart(hierarchy_data)
                    st.plotly_chart(pie_fig, use_container_width=True, key="level_pie_chart")

                # Add sunburst visualization
                st.subheader("Hierarchical Structure")
                sunburst_fig = create_sunburst_chart(hierarchy_data)

                # Try the integer key version
                treemap_chart = treemap_viz(hierarchy_data)
                if treemap_chart is not None:
                    st.plotly_chart(treemap_chart, key="treemap_chart")

                if sunburst_fig is not None:
                    st.plotly_chart(sunburst_fig, use_container_width=True, key="sunburst_chart")
                else:
                    st.info("Not enough hierarchical data to create a sunburst visualization.")

            # Recursive Abstraction Tab
            with viz_tabs[2]:
                st.subheader("Text Abstraction Process")

                # Create abstraction flow visualization
                flow_fig, level_stats = create_abstraction_flow(hierarchy_data)
                if flow_fig:
                    st.plotly_chart(flow_fig, use_container_width=True, key="abstraction_flow")

                # Show content samples across levels
                st.subheader("Content Examples Across Levels")

                # Display example content from each level
                if level_stats:
                    # Create columns for each level
                    cols = st.columns(len(level_stats))

                    # Add content to each column
                    for i, (level, stats) in enumerate(sorted(level_stats.items())):
                        with cols[i]:
                            st.markdown(f"**Level {level}**")
                            st.markdown(f"*{stats['doc_count']} documents*")

                            # Truncate example text
                            example = stats["example"]
                            if len(example) > 300:
                                example = example[:297] + "..."

                            st.text_area("", example, height=200, key=f"example_level_{level}")

                st.info("""
                **Understanding Recursive Abstraction**

                The Raptor algorithm creates progressively more abstract summaries as you move up the hierarchy:

                - **Level 0** contains the original documents or chunks with all details
                - **Level 1** contains summaries of similar documents from Level 0
                - **Level 2** contains "summaries of summaries" from Level 1
                - And so on up to the root level

                Notice how document length typically decreases at higher levels as content becomes more abstract.
                """)

            # Level Details Tab
            # Level Details Tab
            # Level Details Tab
            # Level Details Tab
            with viz_tabs[3]:
                st.subheader("Document Details by Level")

                # Create level selector
                level_options = [f"Level {level}" for level in sorted(
                    [int(level) for level in hierarchy_data["levels"].keys()]
                )]

                if not level_options:
                    st.warning("No levels found in the hierarchy data.")
                else:
                    selected_level = st.selectbox(
                        "Select Level to Explore",
                        options=level_options,
                        key="level_selector"
                    )

                    # Extract level number from selection
                    level_number = int(selected_level.split(" ")[1])


                    # Display documents for this level
                    if level_number in hierarchy_data["levels"]:
                        level_data = hierarchy_data["levels"][level_number]
                        doc_count = level_data["document_count"]

                        st.write(f"**Level {level_number}: {doc_count} documents total**")

                        # Add explanation of what documents at this level represent
                        if level_number == 0:
                            st.info("Level 0 contains the original documents or chunks.")
                        elif level_number == hierarchy_data["metadata"]["tree_depth"]:
                            st.info("This is the root level containing the most abstract summaries.")
                        else:
                            st.info(
                                f"Level {level_number} contains summaries of documents from Level {level_number - 1}.")

                        # Show sample documents
                        sample_docs = level_data.get("sample_documents", [])

                        if not sample_docs:
                            st.warning(f"No sample documents available for Level {level_number}.")
                        else:
                            st.write(f"Showing {len(sample_docs)} sample documents from this level:")

                            # Create a container for document display
                            doc_container = st.container()

                            with doc_container:
                                for i, doc in enumerate(sample_docs):
                                    doc_id = doc['id']
                                    content_preview = doc.get("content", "No content available")

                                    # Get relationships info
                                    has_parent = "parent_id" in doc.get("metadata", {})
                                    parent_id = doc.get("metadata", {}).get("parent_id", "None")
                                    child_count = doc.get("child_count", 0)

                                    # Create expander title with relationship info
                                    title = f"Document {i + 1}: {doc_id[-10:] if len(doc_id) > 10 else doc_id}"
                                    if has_parent:
                                        title += f" (Parent: ...{parent_id[-8:]})"
                                    if child_count:
                                        title += f" (Children: {child_count})"

                                    with st.expander(title):
                                        # Document content section
                                        st.subheader("Content")
                                        st.text_area(
                                            label=f"Document {i + 1} content",
                                            value=content_preview,
                                            height=150,
                                            key=f"doc_content_{level_number}_{i}",
                                            label_visibility="collapsed"
                                        )

                                        # Relationship section
                                        st.subheader("Relationships")
                                        rel_col1, rel_col2 = st.columns(2)

                                        with rel_col1:
                                            if has_parent:
                                                st.info(f"Parent Document: {parent_id}")
                                            else:
                                                st.info("No parent document (top level)")

                                        with rel_col2:
                                            if child_count > 0:
                                                st.info(f"Has {child_count} child documents")
                                            else:
                                                st.info("No child documents")

                                        # Metadata section
                                        st.subheader("Metadata")
                                        if "metadata" in doc and doc["metadata"]:
                                            # Filter out embedding and large arrays
                                            filtered_metadata = {k: v for k, v in doc["metadata"].items()
                                                                 if k not in ["embedding"] and not isinstance(v, (list,
                                                                                                                  np.ndarray))}

                                            # Use a unique container for each JSON output instead of key parameter
                                            json_container = st.container()
                                            with json_container:
                                                st.json(filtered_metadata)  # Remove the key parameter
                                        else:
                                            st.write("No metadata available")

                            # Navigation between levels
                            st.subheader("Navigate Hierarchy")
                            nav_cols = st.columns(3)

                            with nav_cols[0]:
                                if level_number > 0:
                                    if st.button(f"⬇️ Go to Level {level_number - 1} (More detailed)",
                                                 key=f"go_down_{level_number}"):
                                        # This is just a placeholder - in Streamlit we can't directly change the selectbox
                                        # But we can hint the user
                                        st.session_state["level_selector"] = f"Level {level_number - 1}"
                                        st.experimental_rerun()

                            with nav_cols[2]:
                                if level_number < max(int(lvl) for lvl in hierarchy_data["levels"].keys()):
                                    if st.button(f"⬆️ Go to Level {level_number + 1} (More abstract)",
                                                 key=f"go_up_{level_number}"):
                                        st.session_state["level_selector"] = f"Level {level_number + 1}"
                                        st.experimental_rerun()

                    # Option to download hierarchy data
                    st.download_button(
                        label="Download Complete Hierarchy Data as JSON",
                        data=json.dumps(hierarchy_data, indent=2),
                        file_name=f"raptor_hierarchy_{namespace}.json",
                        mime="application/json",
                        key="download_hierarchy_button"
                    )

        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            st.info("Make sure documents have been indexed with Raptor first.")