import streamlit as st
import pandas as pd
import networkx as nx
import json
import plotly.graph_objects as go
import random

st.set_page_config(layout="wide", page_title="Deforestation Stakeholders Network")

st.title("Deforestation Stakeholders and Factors Network")

# Sidebar filters
st.sidebar.header("Network Filters")

# Main function
def main():
    # Upload JSON file
    uploaded_file = st.sidebar.file_uploader("Upload JSON file", type=["json"])
    
    if uploaded_file is not None:
        # Load data from uploaded file
        data = json.loads(uploaded_file.read())
    else:
        # Display warning that no file is uploaded
        st.warning("Please upload a JSON file to visualize the network.")
        
        # Show sample structure
        st.subheader("Expected JSON Structure:")
        st.code('''
{
  "stakeholders": [
    {
      "name": "Stakeholder Name",
      "category": "Category",
      "role": "Role description",
      ...
    }
  ],
  "factors": [
    {
      "Label": "Factor Name",
      "Type": "Factor Type",
      "Description": "Factor description",
      ...
    }
  ],
  "pain_points": [
    {
      "id": "PP-001",
      "category": "Category Name",
      "description": "Description of the pain point",
      ...
    }
  ]
}
        ''')
        return
    
    # Extract stakeholders, factors, and pain points
    stakeholders = data.get('stakeholders', [])
    factors = data.get('factors', [])
    pain_points = data.get('pain_points', [])
    
    # Display data summary
    st.sidebar.subheader("Data Summary")
    st.sidebar.text(f"Stakeholders: {len(stakeholders)}")
    st.sidebar.text(f"Factors: {len(factors)}")
    st.sidebar.text(f"Pain Points: {len(pain_points)}")
    
    # Filter options
    stakeholder_categories = list(set(s.get('category', '') for s in stakeholders))
    factor_types = list(set(f.get('Type', '') for f in factors))
    pain_point_categories = list(set(p.get('category', '') for p in pain_points))
    
    # Show entity types to include
    st.sidebar.subheader("Entity Types")
    include_stakeholders = st.sidebar.checkbox("Include Stakeholders", value=True)
    include_factors = st.sidebar.checkbox("Include Factors", value=True)
    include_pain_points = st.sidebar.checkbox("Include Pain Points", value=True)
    
    # Filters
    if include_stakeholders:
        selected_stakeholder_categories = st.sidebar.multiselect(
            "Filter Stakeholder Categories",
            options=stakeholder_categories,
            default=stakeholder_categories
        )
    else:
        selected_stakeholder_categories = []
    
    if include_factors:
        selected_factor_types = st.sidebar.multiselect(
            "Filter Factor Types",
            options=factor_types,
            default=factor_types
        )
    else:
        selected_factor_types = []
    
    if include_pain_points:
        selected_pain_point_categories = st.sidebar.multiselect(
            "Filter Pain Point Categories",
            options=pain_point_categories,
            default=pain_point_categories
        )
    else:
        selected_pain_point_categories = []
    
    # Connection parameters
    st.sidebar.subheader("Connection Parameters")
    
    stakeholder_connection_prob = st.sidebar.slider(
        "Stakeholder Connection Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    factor_connection_prob = st.sidebar.slider(
        "Factor Connection Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    pain_point_connection_prob = st.sidebar.slider(
        "Pain Point Connection Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    
    keyword_connection_prob = st.sidebar.slider(
        "Keyword Connection Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    random_connection_prob = st.sidebar.slider(
        "Random Connection Probability",
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.01
    )
    
    # Apply filters
    filtered_stakeholders = [s for s in stakeholders if s.get('category', '') in selected_stakeholder_categories]
    filtered_factors = [f for f in factors if f.get('Type', '') in selected_factor_types]
    filtered_pain_points = [p for p in pain_points if p.get('category', '') in selected_pain_point_categories]
    
    # Create a graph
    G = nx.Graph()
    
    # Add stakeholder nodes
    for stakeholder in filtered_stakeholders:
        G.add_node(stakeholder['name'],
                  type='stakeholder',
                  category=stakeholder.get('category', ''),
                  role=stakeholder.get('role', ''))
    
    # Add factor nodes
    for factor in filtered_factors:
        G.add_node(factor['Label'],
                  type='factor',
                  factor_type=factor.get('Type', ''),
                  description=factor.get('Description', ''))
    
    # Add pain point nodes
    for pain_point in filtered_pain_points:
        # Use ID and description to create a unique node name
        node_name = f"PP: {pain_point.get('id', '')}: {pain_point.get('category', '')}"
        G.add_node(node_name,
                  type='pain_point',
                  category=pain_point.get('category', ''),
                  description=pain_point.get('description', ''))
    
    # Helper function to check if two nodes should be connected
    def should_connect(node1_name, node1_data, node2_name, node2_data):
        # Connect stakeholders of the same category
        if node1_data['type'] == 'stakeholder' and node2_data['type'] == 'stakeholder' and \
           node1_data['category'] == node2_data['category']:
            return random.random() < stakeholder_connection_prob
        
        # Connect factors of the same type
        if node1_data['type'] == 'factor' and node2_data['type'] == 'factor' and \
           node1_data['factor_type'] == node2_data['factor_type']:
            return random.random() < factor_connection_prob
        
        # Connect pain points of the same category
        if node1_data['type'] == 'pain_point' and node2_data['type'] == 'pain_point' and \
           node1_data['category'] == node2_data['category']:
            return random.random() < pain_point_connection_prob
        
        # Connect stakeholders to factors based on keywords
        if (node1_data['type'] == 'stakeholder' and node2_data['type'] == 'factor') or \
           (node1_data['type'] == 'factor' and node2_data['type'] == 'stakeholder'):
            
            stakeholder = node1_data if node1_data['type'] == 'stakeholder' else node2_data
            factor = node1_data if node1_data['type'] == 'factor' else node2_data
            stakeholder_name = node1_name if node1_data['type'] == 'stakeholder' else node2_name
            factor_name = node1_name if node1_data['type'] == 'factor' else node2_name
            
            # Check if stakeholder name or role contains words from factor description
            stakeholder_text = (stakeholder_name + ' ' + stakeholder.get('role', '')).lower()
            factor_text = (factor_name + ' ' + factor.get('description', '')).lower()
            
            # List of keywords to check
            keywords = ['deforestation', 'forest', 'mining', 'palm oil', 'regulation', 
                       'government', 'community', 'indigenous', 'environmental', 'agriculture']
            
            # If both texts contain the same keyword, connect them
            for keyword in keywords:
                if keyword in stakeholder_text and keyword in factor_text:
                    return random.random() < keyword_connection_prob
            
            # Add some random connections
            return random.random() < random_connection_prob
        
        # Connect stakeholders to pain points based on keywords
        if (node1_data['type'] == 'stakeholder' and node2_data['type'] == 'pain_point') or \
           (node1_data['type'] == 'pain_point' and node2_data['type'] == 'stakeholder'):
            
            stakeholder = node1_data if node1_data['type'] == 'stakeholder' else node2_data
            pain_point = node1_data if node1_data['type'] == 'pain_point' else node2_data
            stakeholder_name = node1_name if node1_data['type'] == 'stakeholder' else node2_name
            pain_point_name = node1_name if node1_data['type'] == 'pain_point' else node2_name
            
            # Check if stakeholder name or role contains words from pain point description
            stakeholder_text = (stakeholder_name + ' ' + stakeholder.get('role', '')).lower()
            pain_point_text = (pain_point_name + ' ' + pain_point.get('description', '')).lower()
            
            # List of keywords to check (expanded for pain points)
            keywords = ['deforestation', 'forest', 'mining', 'palm oil', 'regulation', 
                        'government', 'community', 'indigenous', 'environmental', 'agriculture',
                        'challenge', 'issue', 'problem', 'conflict', 'risk']
            
            # If both texts contain the same keyword, connect them
            for keyword in keywords:
                if keyword in stakeholder_text and keyword in pain_point_text:
                    return random.random() < keyword_connection_prob
            
            # Add some random connections
            return random.random() < random_connection_prob
        
        # Connect factors to pain points based on keywords
        if (node1_data['type'] == 'factor' and node2_data['type'] == 'pain_point') or \
           (node1_data['type'] == 'pain_point' and node2_data['type'] == 'factor'):
            
            factor = node1_data if node1_data['type'] == 'factor' else node2_data
            pain_point = node1_data if node1_data['type'] == 'pain_point' else node2_data
            factor_name = node1_name if node1_data['type'] == 'factor' else node2_name
            pain_point_name = node1_name if node1_data['type'] == 'pain_point' else node2_name
            
            # Check if factor description contains words from pain point description
            factor_text = (factor_name + ' ' + factor.get('description', '')).lower()
            pain_point_text = (pain_point_name + ' ' + pain_point.get('description', '')).lower()
            
            # List of keywords to check
            keywords = ['deforestation', 'forest', 'mining', 'palm oil', 'regulation', 
                        'government', 'community', 'indigenous', 'environmental', 'agriculture',
                        'challenge', 'issue', 'problem', 'conflict', 'risk']
            
            # If both texts contain the same keyword, connect them
            for keyword in keywords:
                if keyword in factor_text and keyword in pain_point_text:
                    return random.random() < keyword_connection_prob
            
            # Add some random connections
            return random.random() < random_connection_prob
        
        return False
    
    # Add connections between nodes
    for node1 in G.nodes(data=True):
        node1_name, node1_data = node1
        for node2 in G.nodes(data=True):
            node2_name, node2_data = node2
            
            if node1_name != node2_name:  # Avoid self-loops
                if should_connect(node1_name, node1_data, node2_name, node2_data):
                    G.add_edge(node1_name, node2_name, weight=1)
    
    # Calculate positions using a force-directed layout
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    # Create a color map for categories and factor types
    stakeholder_colors = {
        'Influencer': '#1E88E5',
        'Regulator': '#FF0000',
        'Consumer': '#00C853',
        'Supplier': '#FFC107',
        'Partner': '#7B1FA2',
        'Internal': '#795548',
        'Competitor': '#FF5722'
    }
    
    factor_colors = {
        'Direct Driver': '#FF6B6B',
        'Indirect Driver': '#4ECDC4',
        'Exogenous Driver': '#FFD166',
        'Outcome': '#6B5B95'
    }
    
    # Use a distinct color for pain points
    pain_point_color = '#FF00FF'  # Magenta
    
    # Create separate node lists for different visualizations
    stakeholder_nodes = []
    factor_nodes = []
    pain_point_nodes = []
    
    for node, node_data in G.nodes(data=True):
        x, y = pos[node]
        
        node_info = {
            'name': node,
            'x': x,
            'y': y
        }
        
        if node_data['type'] == 'stakeholder':
            node_info['category'] = node_data['category']
            node_info['role'] = node_data['role']
            node_info['color'] = stakeholder_colors.get(node_data['category'], '#999')
            stakeholder_nodes.append(node_info)
        elif node_data['type'] == 'factor':
            node_info['factor_type'] = node_data['factor_type']
            node_info['description'] = node_data['description']
            node_info['color'] = factor_colors.get(node_data['factor_type'], '#999')
            factor_nodes.append(node_info)
        elif node_data['type'] == 'pain_point':
            node_info['category'] = node_data['category']
            node_info['description'] = node_data['description']
            node_info['color'] = pain_point_color
            pain_point_nodes.append(node_info)
    
    # Create edge coordinates
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Build the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(
        go.Scatter(
            x=edge_x, 
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        )
    )
    
    # Add stakeholder nodes by category
    if include_stakeholders:
        for category in stakeholder_categories:
            if category in selected_stakeholder_categories:
                nodes = [n for n in stakeholder_nodes if n['category'] == category]
                if nodes:
                    x = [node['x'] for node in nodes]
                    y = [node['y'] for node in nodes]
                    node_text = [f"<b>{node['name']}</b><br>Category: {node['category']}<br>Role: {node['role']}" for node in nodes]
                    label_text = [node['name'] if len(node['name']) < 20 else node['name'][:17] + "..." for node in nodes]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x, 
                            y=y,
                            mode='markers+text',
                            marker=dict(
                                size=15,
                                color=stakeholder_colors.get(category, '#999'),
                                line=dict(width=1, color='black')
                            ),
                            text=label_text,
                            textposition="top center",
                            hovertext=node_text,
                            hoverinfo="text",
                            name=f"Stakeholder: {category}"
                        )
                    )
    
    # Add factor nodes by type
    if include_factors:
        for factor_type in factor_types:
            if factor_type in selected_factor_types:
                nodes = [n for n in factor_nodes if n['factor_type'] == factor_type]
                if nodes:
                    x = [node['x'] for node in nodes]
                    y = [node['y'] for node in nodes]
                    node_text = [f"<b>{node['name']}</b><br>Type: {node['factor_type']}<br>Description: {node['description']}" for node in nodes]
                    label_text = [node['name'] if len(node['name']) < 20 else node['name'][:17] + "..." for node in nodes]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x, 
                            y=y,
                            mode='markers+text',
                            marker=dict(
                                size=12,
                                color=factor_colors.get(factor_type, '#999'),
                                line=dict(width=1, color='black')
                            ),
                            text=label_text,
                            textposition="top center",
                            hovertext=node_text,
                            hoverinfo="text",
                            name=f"Factor: {factor_type}"
                        )
                    )
    
    # Add pain point nodes by category
    if include_pain_points:
        for category in pain_point_categories:
            if category in selected_pain_point_categories:
                nodes = [n for n in pain_point_nodes if n['category'] == category]
                if nodes:
                    x = [node['x'] for node in nodes]
                    y = [node['y'] for node in nodes]
                    node_text = [f"<b>{node['name']}</b><br>Category: {node['category']}<br>Description: {node['description']}" for node in nodes]
                    label_text = [node['name'] if len(node['name']) < 20 else node['name'][:17] + "..." for node in nodes]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x, 
                            y=y,
                            mode='markers+text',
                            marker=dict(
                                size=13,
                                color=pain_point_color,
                                symbol='diamond',
                                line=dict(width=1, color='black')
                            ),
                            text=label_text,
                            textposition="top center",
                            hovertext=node_text,
                            hoverinfo="text",
                            name=f"Pain Point: {category}"
                        )
                    )
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        legend=dict(
            x=1.05,
            y=0.5
        )
    )
    
    # Add network stats
    st.subheader("Network Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nodes", len(G.nodes()))
    with col2:
        st.metric("Total Connections", len(G.edges()))
    with col3:
        st.metric("Network Density", round(nx.density(G), 4))
    
    # Display the network
    st.plotly_chart(fig, use_container_width=True)
    
    # Create tabs for different data tables
    tab1, tab2, tab3 = st.tabs(["Stakeholders", "Factors", "Pain Points"])
    
    with tab1:
        # Stakeholder table
        st.subheader("Stakeholders")
        
        stakeholder_df = pd.DataFrame([
            {
                "Name": s["name"],
                "Category": s.get("category", ""),
                "Role": s.get("role", ""),
                "Connections": G.degree(s["name"]) if s["name"] in G else 0
            }
            for s in filtered_stakeholders
        ])
        
        st.dataframe(stakeholder_df, use_container_width=True)
    
    with tab2:
        # Factor table
        st.subheader("Factors")
        
        factor_df = pd.DataFrame([
            {
                "Name": f["Label"],
                "Type": f.get("Type", ""),
                "Description": f.get("Description", ""),
                "Connections": G.degree(f["Label"]) if f["Label"] in G else 0
            }
            for f in filtered_factors
        ])
        
        st.dataframe(factor_df, use_container_width=True)
    
    with tab3:
        # Pain point table
        st.subheader("Pain Points")
        
        pain_point_df = pd.DataFrame([
            {
                "ID": p.get("id", ""),
                "Category": p.get("category", ""),
                "Description": p.get("description", ""),
                "Connections": G.degree(f"PP: {p.get('id', '')}: {p.get('category', '')}") if f"PP: {p.get('id', '')}: {p.get('category', '')}" in G else 0
            }
            for p in filtered_pain_points
        ])
        
        st.dataframe(pain_point_df, use_container_width=True)

if __name__ == "__main__":
    main()
