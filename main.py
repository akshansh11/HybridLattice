import streamlit as st
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime
import json
import re

# Page configuration
st.set_page_config(
    page_title="HybridLattice",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #ffebee 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 5px solid #4caf50;
    }
    .reference-box {
        background-color: #fff9c4;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        border-left: 3px solid #fbc02d;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_pattern' not in st.session_state:
    st.session_state.current_pattern = None
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'selected_bending_cell' not in st.session_state:
    st.session_state.selected_bending_cell = 'honeycomb'
if 'selected_stretching_cell' not in st.session_state:
    st.session_state.selected_stretching_cell = 'octet'

# Color scheme for cell types
COLOR_BENDING = 'rgb(135, 206, 250)'  # Sky blue
COLOR_STRETCHING = 'rgb(255, 0, 0)'   # Red

# Unit cell type definitions
BENDING_CELL_TYPES = {
    'honeycomb': {
        'name': 'Honeycomb',
        'description': 'Hexagonal honeycomb structure - high flexibility',
        'energy_absorption': 'High',
        'stiffness': 'Low',
        'applications': 'Impact protection, cushioning'
    },
    'auxetic': {
        'name': 'Auxetic (Re-entrant)',
        'description': 'Re-entrant honeycomb - negative Poisson ratio',
        'energy_absorption': 'Very High',
        'stiffness': 'Very Low',
        'applications': 'Protective equipment, seals'
    },
    'chiral': {
        'name': 'Chiral',
        'description': 'Rotating nodes with tangent ligaments',
        'energy_absorption': 'Medium-High',
        'stiffness': 'Low-Medium',
        'applications': 'Vibration damping, flexible structures'
    },
    'diamond': {
        'name': 'Diamond',
        'description': 'Diamond lattice with long slender members',
        'energy_absorption': 'Medium',
        'stiffness': 'Medium',
        'applications': 'Lightweight panels, moderate loads'
    }
}

STRETCHING_CELL_TYPES = {
    'octet': {
        'name': 'Octet-Truss',
        'description': 'Octahedral-tetrahedral truss - maximum stiffness',
        'energy_absorption': 'Low',
        'stiffness': 'Very High',
        'applications': 'Load-bearing structures, aerospace'
    },
    'cubic': {
        'name': 'Cubic (BCC)',
        'description': 'Body-centered cubic - isotropic properties',
        'energy_absorption': 'Low',
        'stiffness': 'High',
        'applications': 'General structural applications'
    },
    'kelvin': {
        'name': 'Kelvin Cell',
        'description': 'Tetrakaidecahedron - space-filling',
        'energy_absorption': 'Low-Medium',
        'stiffness': 'High',
        'applications': 'Uniform load distribution'
    },
    'pyramidal': {
        'name': 'Pyramidal',
        'description': 'Pyramid lattice with tetrahedral geometry',
        'energy_absorption': 'Low',
        'stiffness': 'Very High',
        'applications': 'Sandwich panels, high-strength cores'
    },
    'isotruss': {
        'name': 'IsoTruss',
        'description': 'Triangulated space-frame - optimal efficiency',
        'energy_absorption': 'Low',
        'stiffness': 'Very High',
        'applications': 'Aerospace, rocket fairings'
    }
}

# ============================================
# UTILITY FUNCTIONS
# ============================================

def configure_gemini_api(api_key):
    """Configure Gemini API with the provided key"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        st.session_state.gemini_model = model
        st.session_state.api_key = api_key
        return True, "API configured successfully!"
    except Exception as e:
        return False, f"Error configuring API: {str(e)}"

def query_gemini_with_references(prompt):
    """Query Gemini and extract references from the response"""
    if st.session_state.gemini_model is None:
        return "Please configure your Gemini API key first.", []
    
    try:
        # Enhanced prompt to request references
        enhanced_prompt = f"""{prompt}

IMPORTANT: At the end of your response, please list any scientific principles, 
studies, or established knowledge you referenced. Format them as:

REFERENCES:
- [Reference 1]: Description
- [Reference 2]: Description
etc.
"""
        
        response = st.session_state.gemini_model.generate_content(enhanced_prompt)
        full_response = response.text
        
        # Extract references
        references = []
        if "REFERENCES:" in full_response:
            ref_section = full_response.split("REFERENCES:")[1]
            ref_lines = [line.strip() for line in ref_section.split('\n') if line.strip().startswith('-')]
            references = [line.strip('- ') for line in ref_lines]
            
            # Remove reference section from main response
            main_response = full_response.split("REFERENCES:")[0].strip()
        else:
            main_response = full_response
            # Try to infer references from content
            if "study" in main_response.lower() or "research" in main_response.lower():
                references.append("General principles from metamaterial mechanics literature")
        
        return main_response, references
        
    except Exception as e:
        return f"Error querying Gemini: {str(e)}", []

def parse_pattern_from_response(response_text, grid_size):
    """Parse pattern matrix from Gemini response"""
    pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
    
    # Look for LAYER_X patterns
    for layer in range(1, grid_size + 1):
        layer_pattern = f"LAYER_{layer}"
        if layer_pattern in response_text:
            # Find the matrix content
            start_idx = response_text.find(layer_pattern)
            chunk = response_text[start_idx:start_idx+500]
            
            # Extract matrix between brackets
            matrix_match = re.search(r'\[(.*?)\]', chunk, re.DOTALL)
            if matrix_match:
                matrix_str = matrix_match.group(1)
                # Parse rows
                rows = [row.strip() for row in matrix_str.split('\n') if row.strip()]
                for i, row in enumerate(rows[:grid_size]):
                    values = [int(x) for x in re.findall(r'\d+', row)]
                    if len(values) == grid_size:
                        pattern[i, :, layer-1] = values
    
    return pattern

def create_honeycomb_cell(center, size, color):
    """Create a honeycomb structure (bending-dominated)"""
    traces = []
    n_sides = 6
    radius = size / 2.5
    
    # Vertical beams
    for i in range(n_sides):
        angle = i * 2 * np.pi / n_sides
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        
        traces.append(go.Scatter3d(
            x=[x, x],
            y=[y, y],
            z=[center[2] - size/2, center[2] + size/2],
            mode='lines',
            line=dict(color=color, width=8),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Horizontal hexagonal rings
    heights = [-size/2, -size/4, 0, size/4, size/2]
    for h in heights:
        for i in range(n_sides):
            angle1 = i * 2 * np.pi / n_sides
            angle2 = (i + 1) * 2 * np.pi / n_sides
            
            x1 = center[0] + radius * np.cos(angle1)
            y1 = center[1] + radius * np.sin(angle1)
            x2 = center[0] + radius * np.cos(angle2)
            y2 = center[1] + radius * np.sin(angle2)
            
            traces.append(go.Scatter3d(
                x=[x1, x2],
                y=[y1, y2],
                z=[center[2] + h, center[2] + h],
                mode='lines',
                line=dict(color=color, width=8),
                hoverinfo='skip',
                showlegend=False
            ))
    
    return traces

def create_auxetic_cell(center, size, color):
    """Create an auxetic (re-entrant) structure"""
    traces = []
    r_outer = size / 2.8
    r_inner = size / 5
    angle_offset = np.pi / 6
    
    # Re-entrant honeycomb structure
    heights = [-size/2, -size/4, 0, size/4, size/2]
    
    for h in heights:
        for i in range(6):
            angle = i * np.pi / 3
            
            # Outer vertices
            x_out1 = center[0] + r_outer * np.cos(angle)
            y_out1 = center[1] + r_outer * np.sin(angle)
            
            # Inner vertices (re-entrant)
            x_in = center[0] + r_inner * np.cos(angle + angle_offset)
            y_in = center[1] + r_inner * np.sin(angle + angle_offset)
            
            # Connect outer to inner
            traces.append(go.Scatter3d(
                x=[x_out1, x_in],
                y=[y_out1, y_in],
                z=[center[2] + h, center[2] + h],
                mode='lines',
                line=dict(color=color, width=7),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Vertical connections
    for i in range(6):
        angle = i * np.pi / 3
        x = center[0] + r_outer * np.cos(angle)
        y = center[1] + r_outer * np.sin(angle)
        
        traces.append(go.Scatter3d(
            x=[x, x],
            y=[y, y],
            z=[center[2] - size/2, center[2] + size/2],
            mode='lines',
            line=dict(color=color, width=7),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def create_chiral_cell(center, size, color):
    """Create a chiral structure with rotating nodes"""
    traces = []
    n_nodes = 4
    radius = size / 3.5
    node_size = size / 12
    
    # Create chiral nodes at different heights
    heights = [-size/2, 0, size/2]
    
    for idx, h in enumerate(heights):
        rotation = idx * np.pi / 8  # Rotate each layer
        
        for i in range(n_nodes):
            angle = i * 2 * np.pi / n_nodes + rotation
            angle_next = (i + 1) * 2 * np.pi / n_nodes + rotation
            
            x1 = center[0] + radius * np.cos(angle)
            y1 = center[1] + radius * np.sin(angle)
            x2 = center[0] + radius * np.cos(angle_next)
            y2 = center[1] + radius * np.sin(angle_next)
            
            # Tangent connections
            mid_angle = angle + np.pi / (2 * n_nodes)
            x_mid = center[0] + (radius - node_size) * np.cos(mid_angle)
            y_mid = center[1] + (radius - node_size) * np.sin(mid_angle)
            
            traces.append(go.Scatter3d(
                x=[x1, x_mid],
                y=[y1, y_mid],
                z=[center[2] + h, center[2] + h],
                mode='lines',
                line=dict(color=color, width=7),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Vertical connections between layers
    for i in range(n_nodes):
        for h_idx in range(len(heights) - 1):
            angle1 = i * 2 * np.pi / n_nodes + h_idx * np.pi / 8
            angle2 = i * 2 * np.pi / n_nodes + (h_idx + 1) * np.pi / 8
            
            x1 = center[0] + radius * np.cos(angle1)
            y1 = center[1] + radius * np.sin(angle1)
            x2 = center[0] + radius * np.cos(angle2)
            y2 = center[1] + radius * np.sin(angle2)
            
            traces.append(go.Scatter3d(
                x=[x1, x2],
                y=[y1, y2],
                z=[center[2] + heights[h_idx], center[2] + heights[h_idx + 1]],
                mode='lines',
                line=dict(color=color, width=6),
                hoverinfo='skip',
                showlegend=False
            ))
    
    return traces

def create_diamond_cell(center, size, color):
    """Create a diamond lattice structure"""
    traces = []
    s = size / 2
    
    # Diamond lattice vertices
    vertices = np.array([
        [0, 0, 0],
        [s, s, 0],
        [s, 0, s],
        [0, s, s],
        [-s, -s, 0],
        [-s, 0, -s],
        [0, -s, -s],
        [s, -s, 0],
        [-s, 0, s],
        [0, -s, s]
    ]) + center
    
    # Diamond connections (slender beams)
    connections = [
        (0, 1), (0, 2), (0, 3),
        (0, 4), (0, 5), (0, 6),
        (1, 2), (2, 3), (3, 1),
        (4, 5), (5, 6), (6, 4),
        (7, 1), (8, 3), (9, 2)
    ]
    
    for conn in connections:
        if conn[0] < len(vertices) and conn[1] < len(vertices):
            p1, p2 = vertices[conn[0]], vertices[conn[1]]
            traces.append(go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode='lines',
                line=dict(color=color, width=6),
                hoverinfo='skip',
                showlegend=False
            ))
    
    return traces

def create_octet_truss_cell(center, size, color):
    """Create an octet-truss structure (stretching-dominated)"""
    traces = []
    
    # Define vertices relative to center
    s = size / 2
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Bottom
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],      # Top
        [0, 0, -s], [0, 0, s],                                # Face centers z
        [-s, 0, 0], [s, 0, 0],                                # Face centers x
        [0, -s, 0], [0, s, 0],                                # Face centers y
        [0, 0, 0]                                             # Center
    ]) + center
    
    # Enhanced connections for octet-truss
    connections = [
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
        # Bottom face to bottom center
        (0, 8), (1, 8), (2, 8), (3, 8),
        # Top face to top center
        (4, 9), (5, 9), (6, 9), (7, 9),
        # Face centers to cell center
        (8, 14), (9, 14), (10, 14), (11, 14), (12, 14), (13, 14),
        # Cell center to corners
        (14, 0), (14, 1), (14, 2), (14, 3),
        (14, 4), (14, 5), (14, 6), (14, 7),
        # Additional diagonal bracing
        (0, 12), (1, 12), (2, 11), (3, 10),
        (4, 12), (5, 12), (6, 11), (7, 10),
    ]
    
    for conn in connections:
        p1, p2 = vertices[conn[0]], vertices[conn[1]]
        traces.append(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color=color, width=10),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def create_cubic_bcc_cell(center, size, color):
    """Create a Body-Centered Cubic (BCC) structure"""
    traces = []
    s = size / 2
    
    # BCC vertices
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Bottom corners
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],      # Top corners
        [0, 0, 0]                                              # Body center
    ]) + center
    
    # BCC connections - all corners to center
    connections = [
        (0, 8), (1, 8), (2, 8), (3, 8),
        (4, 8), (5, 8), (6, 8), (7, 8),
        # Edge connections for stability
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical
    ]
    
    for conn in connections:
        p1, p2 = vertices[conn[0]], vertices[conn[1]]
        traces.append(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color=color, width=9),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def create_kelvin_cell(center, size, color):
    """Create a Kelvin cell (tetrakaidecahedron) structure"""
    traces = []
    s = size / 2.5
    
    # Kelvin cell vertices (14-faced polyhedron)
    vertices = np.array([
        # Square faces (6 vertices on each axis)
        [-s, 0, 0], [s, 0, 0],
        [0, -s, 0], [0, s, 0],
        [0, 0, -s], [0, 0, s],
        # Hexagonal faces (8 vertices at cube corners, scaled)
        [-s*0.7, -s*0.7, -s*0.7], [s*0.7, -s*0.7, -s*0.7],
        [s*0.7, s*0.7, -s*0.7], [-s*0.7, s*0.7, -s*0.7],
        [-s*0.7, -s*0.7, s*0.7], [s*0.7, -s*0.7, s*0.7],
        [s*0.7, s*0.7, s*0.7], [-s*0.7, s*0.7, s*0.7],
    ]) + center
    
    # Kelvin cell connections
    connections = [
        # Square face connections
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 4), (2, 5), (3, 4), (3, 5),
        # Hexagonal vertex connections
        (6, 7), (7, 8), (8, 9), (9, 6),  # Bottom hex
        (10, 11), (11, 12), (12, 13), (13, 10),  # Top hex
        (6, 10), (7, 11), (8, 12), (9, 13),  # Vertical
        # Connect to face centers
        (0, 6), (0, 9), (0, 10), (0, 13),
        (1, 7), (1, 8), (1, 11), (1, 12),
    ]
    
    for conn in connections:
        p1, p2 = vertices[conn[0]], vertices[conn[1]]
        traces.append(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color=color, width=8),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def create_pyramidal_cell(center, size, color):
    """Create a pyramidal lattice structure"""
    traces = []
    s = size / 2
    
    # Pyramidal vertices
    vertices = np.array([
        # Bottom base
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        # Top base  
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        # Pyramid apex points (middle of each face)
        [0, 0, 0],  # Center
        [0, 0, -s], [0, 0, s],  # Z-faces
        [-s, 0, 0], [s, 0, 0],  # X-faces
        [0, -s, 0], [0, s, 0],  # Y-faces
    ]) + center
    
    # Pyramidal connections - tetrahedrons
    connections = [
        # Bottom pyramids
        (0, 9), (1, 9), (2, 9), (3, 9),
        (9, 13), (9, 14),
        # Top pyramids
        (4, 10), (5, 10), (6, 10), (7, 10),
        (10, 13), (10, 14),
        # Side pyramids
        (0, 11), (3, 11), (4, 11), (7, 11),
        (1, 12), (2, 12), (5, 12), (6, 12),
        (0, 13), (1, 13), (4, 13), (5, 13),
        (2, 14), (3, 14), (6, 14), (7, 14),
        # Center connections
        (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14),
    ]
    
    for conn in connections:
        p1, p2 = vertices[conn[0]], vertices[conn[1]]
        traces.append(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color=color, width=9),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def create_isotruss_cell(center, size, color):
    """Create an IsoTruss structure (triangulated space-frame)"""
    traces = []
    s = size / 2
    
    # IsoTruss vertices - highly triangulated
    vertices = np.array([
        # Outer vertices
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        # Mid-edge vertices
        [0, -s, -s], [s, 0, -s], [0, s, -s], [-s, 0, -s],  # Bottom edges
        [0, -s, s], [s, 0, s], [0, s, s], [-s, 0, s],      # Top edges
        [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0],    # Middle edges
    ]) + center
    
    # IsoTruss connections - maximize triangulation
    connections = [
        # Bottom triangulation
        (0, 8), (8, 1), (1, 9), (9, 2), (2, 10), (10, 3), (3, 11), (11, 0),
        (8, 9), (9, 10), (10, 11), (11, 8),
        # Top triangulation
        (4, 12), (12, 5), (5, 13), (13, 6), (6, 14), (14, 7), (7, 15), (15, 4),
        (12, 13), (13, 14), (14, 15), (15, 12),
        # Vertical triangulation
        (0, 16), (16, 4), (1, 17), (17, 5), (2, 18), (18, 6), (3, 19), (19, 7),
        (16, 17), (17, 18), (18, 19), (19, 16),
        # Diagonal bracing
        (8, 16), (9, 17), (10, 18), (11, 19),
        (12, 16), (13, 17), (14, 18), (15, 19),
        # Cross-diagonal bracing
        (0, 17), (1, 16), (2, 19), (3, 18),
        (4, 17), (5, 16), (6, 19), (7, 18),
    ]
    
    for conn in connections:
        p1, p2 = vertices[conn[0]], vertices[conn[1]]
        traces.append(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color=color, width=8),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def visualize_3d_pattern(pattern, cell_size=1.0, show_legend=True):
    """Create 3D visualization of the pattern"""
    grid_size = pattern.shape[0]
    fig = go.Figure()
    
    bending_added = False
    stretching_added = False
    
    # Get selected cell types
    bending_type = st.session_state.selected_bending_cell
    stretching_type = st.session_state.selected_stretching_cell
    
    # Cell type function mapping
    bending_functions = {
        'honeycomb': create_honeycomb_cell,
        'auxetic': create_auxetic_cell,
        'chiral': create_chiral_cell,
        'diamond': create_diamond_cell
    }
    
    stretching_functions = {
        'octet': create_octet_truss_cell,
        'cubic': create_cubic_bcc_cell,
        'kelvin': create_kelvin_cell,
        'pyramidal': create_pyramidal_cell,
        'isotruss': create_isotruss_cell
    }
    
    # Generate lattice structures
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                center = np.array([i * cell_size, j * cell_size, k * cell_size])
                
                if pattern[i, j, k] == 1:  # Bending-dominated
                    create_func = bending_functions.get(bending_type, create_honeycomb_cell)
                    traces = create_func(center, cell_size, COLOR_BENDING)
                    for trace in traces:
                        if not bending_added and show_legend:
                            trace.showlegend = True
                            trace.name = f'Bending: {BENDING_CELL_TYPES[bending_type]["name"]}'
                            bending_added = True
                        fig.add_trace(trace)
                else:  # Stretching-dominated
                    create_func = stretching_functions.get(stretching_type, create_octet_truss_cell)
                    traces = create_func(center, cell_size, COLOR_STRETCHING)
                    for trace in traces:
                        if not stretching_added and show_legend:
                            trace.showlegend = True
                            trace.name = f'Stretching: {STRETCHING_CELL_TYPES[stretching_type]["name"]}'
                            stretching_added = True
                        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(title='Y', backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(title='Z', backgroundcolor="white", gridcolor="lightgray"),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        showlegend=show_legend,
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_2d_layer_view(pattern):
    """Create 2D views of each layer"""
    grid_size = pattern.shape[0]
    
    fig = go.Figure()
    
    # Create subplots for each layer
    for layer in range(grid_size):
        layer_data = pattern[:, :, layer]
        
        # Create custom colorscale (1=blue, 2=red)
        colorscale = [[0, COLOR_BENDING], [1, COLOR_STRETCHING]]
        
        fig.add_trace(go.Heatmap(
            z=layer_data,
            colorscale=colorscale,
            showscale=False,
            text=np.where(layer_data == 1, 'B', 'S'),
            texttemplate='%{text}',
            textfont=dict(size=16, color='white'),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Type: %{text}<extra></extra>',
            name=f'Layer {layer+1}'
        ))
    
    # Update layout for slider
    fig.update_layout(
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': 0,
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Layer: ',
                'visible': True,
                'xanchor': 'right'
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'steps': [
                {
                    'args': [{'visible': [i == j for j in range(grid_size)]}],
                    'label': str(i+1),
                    'method': 'update'
                }
                for i in range(grid_size)
            ]
        }],
        height=500,
        yaxis=dict(scaleanchor='x', scaleratio=1),
        title='2D Layer View (Slide to view different layers)'
    )
    
    # Set initial visibility
    for i, trace in enumerate(fig.data):
        trace.visible = (i == 0)
    
    return fig

def generate_pattern_presets(grid_size):
    """Generate preset patterns"""
    presets = {}
    
    # Checkerboard
    pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if (i + j + k) % 2 == 0:
                    pattern[i, j, k] = 2
    presets['Checkerboard'] = pattern
    
    # Core-Shell
    pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # Outer shell is stretching-dominated
                if i == 0 or i == grid_size-1 or j == 0 or j == grid_size-1 or k == 0 or k == grid_size-1:
                    pattern[i, j, k] = 2
    presets['Core-Shell'] = pattern
    
    # Layered
    pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
    for k in range(grid_size):
        if k < grid_size // 2:
            pattern[:, :, k] = 2
    presets['Layered (Bottom Strong)'] = pattern
    
    # Sandwich
    pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
    pattern[:, :, 0] = 2
    pattern[:, :, -1] = 2
    presets['Sandwich'] = pattern
    
    # Gradient
    pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
    for k in range(grid_size):
        if np.random.rand() < k / grid_size:
            pattern[:, :, k] = 2
    presets['Gradient'] = pattern
    
    return presets

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown('<div class="main-header">Hybrid Architected Material Designer<br>Powered by Agentic AI</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
        # API Key input
        st.markdown("### Gemini API Key")
        api_key_input = st.text_input(
            "Enter API Key",
            type="password",
            value=st.session_state.api_key if st.session_state.api_key else "",
            help="Get your free API key from https://makersuite.google.com/app/apikey"
        )
        
        if st.button("Configure API"):
            if api_key_input:
                success, message = configure_gemini_api(api_key_input)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter an API key")
        
        st.markdown("---")
        
        # Grid size selection
        st.markdown("### Grid Configuration")
        grid_size = st.selectbox(
            "Grid Size",
            options=[2, 4],
            index=1,
            help="Select 2x2x2 or 4x4x4 grid"
        )
        
        # Unit cell selection
        st.markdown("### Unit Cell Types")
        
        st.markdown("#### Bending-Dominated (Sky Blue)")
        selected_bending = st.selectbox(
            "Select Bending Cell Type",
            options=list(BENDING_CELL_TYPES.keys()),
            format_func=lambda x: BENDING_CELL_TYPES[x]['name'],
            index=list(BENDING_CELL_TYPES.keys()).index(st.session_state.selected_bending_cell),
            key='bending_selector'
        )
        
        if selected_bending != st.session_state.selected_bending_cell:
            st.session_state.selected_bending_cell = selected_bending
        
        # Show bending cell info
        bending_info = BENDING_CELL_TYPES[selected_bending]
        st.markdown(f'<div style="background-color: {COLOR_BENDING}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">'
                   f'<strong>{bending_info["name"]}</strong><br>'
                   f'<small>{bending_info["description"]}</small><br>'
                   f'<small>Stiffness: {bending_info["stiffness"]}</small><br>'
                   f'<small>Energy Absorption: {bending_info["energy_absorption"]}</small>'
                   f'</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("#### Stretching-Dominated (Red)")
        selected_stretching = st.selectbox(
            "Select Stretching Cell Type",
            options=list(STRETCHING_CELL_TYPES.keys()),
            format_func=lambda x: STRETCHING_CELL_TYPES[x]['name'],
            index=list(STRETCHING_CELL_TYPES.keys()).index(st.session_state.selected_stretching_cell),
            key='stretching_selector'
        )
        
        if selected_stretching != st.session_state.selected_stretching_cell:
            st.session_state.selected_stretching_cell = selected_stretching
        
        # Show stretching cell info
        stretching_info = STRETCHING_CELL_TYPES[selected_stretching]
        st.markdown(f'<div style="background-color: {COLOR_STRETCHING}; padding: 10px; border-radius: 5px; color: white;">'
                   f'<strong>{stretching_info["name"]}</strong><br>'
                   f'<small>{stretching_info["description"]}</small><br>'
                   f'<small>Stiffness: {stretching_info["stiffness"]}</small><br>'
                   f'<small>Energy Absorption: {stretching_info["energy_absorption"]}</small>'
                   f'</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Preset patterns
        st.markdown("### Preset Patterns")
        presets = generate_pattern_presets(grid_size)
        preset_choice = st.selectbox("Select Preset", options=list(presets.keys()))
        
        if st.button("Load Preset"):
            st.session_state.current_pattern = presets[preset_choice]
            st.success(f"Loaded {preset_choice} pattern!")
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.current_pattern is not None:
            st.markdown("### Pattern Statistics")
            pattern = st.session_state.current_pattern
            total_cells = pattern.size
            bending_cells = np.sum(pattern == 1)
            stretching_cells = np.sum(pattern == 2)
            
            st.metric("Total Cells", total_cells)
            st.metric("Bending Cells", f"{bending_cells} ({100*bending_cells/total_cells:.1f}%)")
            st.metric("Stretching Cells", f"{stretching_cells} ({100*stretching_cells/total_cells:.1f}%)")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Design Studio", 
        "AI Design Assistant", 
        "Chat with AI",
        "Documentation"
    ])
    
    # TAB 1: Design Studio
    with tab1:
        st.markdown("## Interactive Design Studio")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Manual Pattern Editor")
            
            if st.session_state.current_pattern is not None:
                pattern = st.session_state.current_pattern
                
                # Layer selector
                layer_to_edit = st.slider("Select Layer to Edit", 0, grid_size-1, 0)
                
                st.write(f"**Layer {layer_to_edit + 1}** (Click cells to toggle)")
                
                # Create editable grid
                layer_data = pattern[:, :, layer_to_edit].copy()
                
                # Display as buttons in a grid
                for i in range(grid_size):
                    cols = st.columns(grid_size)
                    for j in range(grid_size):
                        with cols[j]:
                            cell_type = layer_data[i, j]
                            color = COLOR_BENDING if cell_type == 1 else COLOR_STRETCHING
                            label = "B" if cell_type == 1 else "S"
                            
                            if st.button(
                                label,
                                key=f"cell_{layer_to_edit}_{i}_{j}",
                                help=f"Cell ({i},{j}) - Click to toggle"
                            ):
                                # Toggle cell type
                                pattern[i, j, layer_to_edit] = 1 if pattern[i, j, layer_to_edit] == 2 else 2
                                st.session_state.current_pattern = pattern
                                st.rerun()
                
                st.markdown("---")
                
                # Action buttons
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Reset Pattern"):
                        st.session_state.current_pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
                        st.rerun()
                
                with col_b:
                    if st.button("Randomize"):
                        st.session_state.current_pattern = np.random.choice([1, 2], size=(grid_size, grid_size, grid_size))
                        st.rerun()
            
            else:
                st.info("Load a preset pattern from the sidebar to begin editing")
        
        with col2:
            st.markdown("### Visualization")
            
            if st.session_state.current_pattern is not None:
                viz_type = st.radio(
                    "Visualization Type",
                    options=["3D Interactive", "2D Layer View"],
                    horizontal=True
                )
                
                if viz_type == "3D Interactive":
                    with st.spinner("Generating 3D visualization..."):
                        fig = visualize_3d_pattern(st.session_state.current_pattern)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("Use mouse to rotate, zoom, and pan the 3D model")
                
                else:
                    fig = create_2d_layer_view(st.session_state.current_pattern)
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("No pattern loaded. Load a preset or create a new pattern.")
    
    # TAB 2: AI Design Assistant
    with tab2:
        st.markdown("## AI-Powered Design Generation")
        
        if st.session_state.api_key is None:
            st.warning("Please configure your Gemini API key in the sidebar first.")
        else:
            st.markdown('<div class="info-box">Use AI to generate optimal patterns based on your requirements</div>', 
                       unsafe_allow_html=True)
            
            # Design task selection
            task = st.selectbox(
                "Select AI Task",
                options=[
                    "Generate Optimal Pattern",
                    "Analyze Current Pattern",
                    "Property Prediction",
                    "Design Comparison",
                    "Creative Pattern Generation"
                ]
            )
            
            st.markdown("---")
            
            if task == "Generate Optimal Pattern":
                st.markdown("### Design Requirements")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    loading_direction = st.selectbox(
                        "Primary Loading Direction",
                        options=["Z-axis (Compression)", "X-axis", "Y-axis", "Multi-directional"]
                    )
                    
                    stiffness_req = st.slider("Stiffness Requirement", 1, 10, 7)
                
                with col2:
                    energy_absorption = st.slider("Energy Absorption (%)", 0, 100, 30)
                    weight_importance = st.slider("Weight Minimization", 1, 10, 5)
                
                additional_req = st.text_area(
                    "Additional Requirements (Optional)",
                    placeholder="E.g., symmetric design, specific failure mode, cost constraints..."
                )
                
                if st.button("Generate Design", type="primary"):
                    with st.spinner("AI is generating optimal design..."):
                        # Create prompt
                        prompt = f"""You are an expert in mechanical metamaterials and lattice structures.
Design an optimal {grid_size}x{grid_size}x{grid_size} hybrid architectured material.

REQUIREMENTS:
- Primary loading direction: {loading_direction}
- Stiffness requirement: {stiffness_req}/10
- Energy absorption: {energy_absorption}%
- Weight minimization importance: {weight_importance}/10
- Additional requirements: {additional_req if additional_req else 'None'}

CELL TYPES:
- Type 1 = Bending-dominated (honeycomb, flexible, absorbs energy)
- Type 2 = Stretching-dominated (octet-truss, stiff, load-bearing)

PROVIDE YOUR ANSWER IN THIS EXACT FORMAT:

LAYER_1:
[1 2 1 2
 2 1 2 1
 1 2 1 2
 2 1 2 1]

LAYER_2:
[values here]

(Continue for all {grid_size} layers)

EXPLANATION:
[Your design rationale in 2-3 sentences]

EXPECTED PERFORMANCE:
[Predicted mechanical behavior]

Use only 1 and 2 in the matrices. Show all {grid_size} layers."""
                        
                        response, references = query_gemini_with_references(prompt)
                        
                        # Display response
                        st.markdown("### AI Design Response")
                        st.markdown(response)
                        
                        # Display references
                        if references:
                            st.markdown('<div class="reference-box"><strong>References:</strong>', unsafe_allow_html=True)
                            for ref in references:
                                st.markdown(f"- {ref}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Parse and visualize
                        try:
                            pattern = parse_pattern_from_response(response, grid_size)
                            st.session_state.current_pattern = pattern
                            
                            st.success("Pattern extracted and loaded!")
                            
                            # Show 3D visualization
                            st.markdown("### 3D Visualization")
                            fig = visualize_3d_pattern(pattern)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Could not parse pattern: {str(e)}")
            
            elif task == "Analyze Current Pattern":
                if st.session_state.current_pattern is None:
                    st.warning("Please load or create a pattern first")
                else:
                    if st.button("Analyze Pattern", type="primary"):
                        with st.spinner("Analyzing pattern..."):
                            pattern = st.session_state.current_pattern
                            
                            prompt = f"""Analyze this {grid_size}x{grid_size}x{grid_size} metamaterial pattern:

PATTERN DATA:
Layer 1: {pattern[:,:,0].tolist()}
{f"Layer 2: {pattern[:,:,1].tolist()}" if grid_size > 1 else ""}
{f"Layer 3: {pattern[:,:,2].tolist()}" if grid_size > 2 else ""}
{f"Layer 4: {pattern[:,:,3].tolist()}" if grid_size > 3 else ""}

Where 1 = bending-dominated (flexible) and 2 = stretching-dominated (stiff)

PROVIDE:
1. Stiffness analysis (X, Y, Z directions) - rate 1-10 for each
2. Energy absorption capability - Low/Medium/High with reasoning
3. Expected failure mode and location
4. Load distribution characteristics
5. Recommended applications (at least 3)
6. Design improvement suggestions (at least 2)

Be specific and provide numerical estimates where possible."""
                            
                            response, references = query_gemini_with_references(prompt)
                            
                            st.markdown("### Analysis Results")
                            st.markdown(response)
                            
                            if references:
                                st.markdown('<div class="reference-box"><strong>References:</strong>', unsafe_allow_html=True)
                                for ref in references:
                                    st.markdown(f"- {ref}")
                                st.markdown('</div>', unsafe_allow_html=True)
            
            elif task == "Property Prediction":
                if st.session_state.current_pattern is None:
                    st.warning("Please load or create a pattern first")
                else:
                    if st.button("Predict Properties", type="primary"):
                        with st.spinner("Predicting mechanical properties..."):
                            pattern = st.session_state.current_pattern
                            
                            prompt = f"""As a mechanical engineering expert, predict detailed properties of this {grid_size}x{grid_size}x{grid_size} pattern:

PATTERN: (showing middle layer)
{pattern[:,:,grid_size//2].tolist()}

Where 1=flexible honeycomb, 2=stiff octet-truss

PREDICT WITH NUMERICAL ESTIMATES:
1. Effective Young's Modulus (Ex, Ey, Ez) - provide approximate GPa values
2. Poisson's ratios
3. Energy absorption capacity (J/cm³)
4. Relative density (%)
5. Failure stress under compression (MPa)
6. Specific stiffness (strength-to-weight ratio)
7. Best applications with justification

Base your estimates on typical properties:
- Honeycomb: E ≈ 0.1-1 GPa, flexible
- Octet-truss: E ≈ 5-50 GPa, stiff"""
                            
                            response, references = query_gemini_with_references(prompt)
                            
                            st.markdown("### Property Predictions")
                            st.markdown(response)
                            
                            if references:
                                st.markdown('<div class="reference-box"><strong>References:</strong>', unsafe_allow_html=True)
                                for ref in references:
                                    st.markdown(f"- {ref}")
                                st.markdown('</div>', unsafe_allow_html=True)
            
            elif task == "Design Comparison":
                st.markdown("### Compare Multiple Patterns")
                
                if st.button("Generate Comparison Patterns", type="primary"):
                    with st.spinner("Generating patterns for comparison..."):
                        presets = generate_pattern_presets(grid_size)
                        
                        # Display patterns
                        cols = st.columns(3)
                        pattern_names = list(presets.keys())[:3]
                        
                        for idx, (col, name) in enumerate(zip(cols, pattern_names)):
                            with col:
                                st.markdown(f"**{name}**")
                                fig = create_2d_layer_view(presets[name])
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # AI comparison
                        prompt = f"""Compare these three {grid_size}x{grid_size}x{grid_size} patterns:

PATTERN A ({pattern_names[0]}):
{presets[pattern_names[0]][:,:,0].tolist()}

PATTERN B ({pattern_names[1]}):
{presets[pattern_names[1]][:,:,0].tolist()}

PATTERN C ({pattern_names[2]}):
{presets[pattern_names[2]][:,:,0].tolist()}

COMPARE AND RANK FOR:
1. Compressive strength (1st, 2nd, 3rd) with reasoning
2. Energy absorption (1st, 2nd, 3rd) with reasoning
3. Weight efficiency (1st, 2nd, 3rd) with reasoning
4. Manufacturing complexity
5. Cost-effectiveness

Provide specific mechanical reasoning for each ranking."""
                        
                        response, references = query_gemini_with_references(prompt)
                        
                        st.markdown("### Comparison Results")
                        st.markdown(response)
                        
                        if references:
                            st.markdown('<div class="reference-box"><strong>References:</strong>', unsafe_allow_html=True)
                            for ref in references:
                                st.markdown(f"- {ref}")
                            st.markdown('</div>', unsafe_allow_html=True)
            
            elif task == "Creative Pattern Generation":
                st.markdown("### Generate Creative Patterns")
                
                creativity_prompt = st.text_area(
                    "Describe Your Creative Vision",
                    placeholder="E.g., 'spiral pattern with increasing stiffness', 'nature-inspired honeycomb gradient', 'chess board with strategic reinforcement'...",
                    height=100
                )
                
                if st.button("Generate Creative Design", type="primary"):
                    if creativity_prompt:
                        with st.spinner("AI is being creative..."):
                            prompt = f"""Design a creative {grid_size}x{grid_size}x{grid_size} metamaterial pattern based on this vision:

"{creativity_prompt}"

Be innovative! Think about:
- Artistic patterns (spirals, waves, fractals)
- Functional gradients (variable stiffness)
- Nature-inspired designs
- Geometric patterns

PROVIDE:
1. Creative pattern name
2. All {grid_size} layers in LAYER_X format
3. Design inspiration
4. Mechanical behavior prediction
5. Recommended applications

Use 1 for flexible cells and 2 for stiff cells."""
                            
                            response, references = query_gemini_with_references(prompt)
                            
                            st.markdown("### Creative Design")
                            st.markdown(response)
                            
                            if references:
                                st.markdown('<div class="reference-box"><strong>References:</strong>', unsafe_allow_html=True)
                                for ref in references:
                                    st.markdown(f"- {ref}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Try to parse and visualize
                            try:
                                pattern = parse_pattern_from_response(response, grid_size)
                                st.session_state.current_pattern = pattern
                                
                                st.success("Creative pattern loaded!")
                                
                                fig = visualize_3d_pattern(pattern)
                                st.plotly_chart(fig, use_container_width=True)
                            except:
                                st.info("Extract the pattern manually if needed")
                    else:
                        st.warning("Please describe your creative vision")
    
    # TAB 3: Chat with AI
    with tab3:
        st.markdown("## Chat with Material Design Expert")
        
        if st.session_state.api_key is None:
            st.warning("Please configure your Gemini API key in the sidebar first.")
        else:
            st.markdown('<div class="info-box">Ask questions about metamaterials, get design advice, or discuss mechanical properties</div>', 
                       unsafe_allow_html=True)
            
            # Display chat history
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{message["content"]}</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message assistant-message"><strong>AI Assistant:</strong><br>{message["content"]}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Show references if available
                        if 'references' in message and message['references']:
                            st.markdown('<div class="reference-box"><strong>References:</strong>', unsafe_allow_html=True)
                            for ref in message['references']:
                                st.markdown(f"- {ref}")
                            st.markdown('</div>', unsafe_allow_html=True)
            
            # Chat input
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Your question:",
                    placeholder="Ask about metamaterials, design strategies, material properties, or get design recommendations...",
                    height=100,
                    key="chat_input"
                )
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    submit = st.form_submit_button("Send", type="primary")
                with col2:
                    clear = st.form_submit_button("Clear Chat")
                
                if clear:
                    st.session_state.chat_history = []
                    st.rerun()
                
                if submit and user_input:
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    # Get AI response
                    with st.spinner("AI is thinking..."):
                        context_prompt = f"""You are an expert in mechanical metamaterials, lattice structures, 
and hybrid architectured materials. Answer the following question with detailed technical knowledge.

Question: {user_input}

Provide a comprehensive answer with:
1. Direct answer to the question
2. Technical details and explanations
3. Practical examples if relevant
4. Design recommendations if applicable"""
                        
                        response, references = query_gemini_with_references(context_prompt)
                        
                        # Add AI response
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response,
                            'references': references
                        })
                    
                    st.rerun()
    
    # TAB 4: Documentation
    with tab4:
        st.markdown("## Documentation and User Guide")
        
        st.markdown("""
        ### Overview
        
        The Hybrid Architectured Material Designer is an interactive application that combines 
        AI-powered design generation with manual editing capabilities for creating optimized 
        lattice structures.
        
        ### Unit Cell Types
        
        **1. Bending-Dominated (Sky Blue)**
        - Structure: Hexagonal honeycomb
        - Properties: Flexible, compliant
        - Advantages: High energy absorption, lightweight
        - Applications: Impact protection, cushioning
        
        **2. Stretching-Dominated (Red)**
        - Structure: Octet-truss
        - Properties: Stiff, strong
        - Advantages: High load-bearing capacity, structural integrity
        - Applications: Load-bearing structures, aerospace components
        
        ### Features
        
        #### 1. Design Studio
        - **Manual Editor**: Click individual cells to toggle between bending and stretching types
        - **3D Visualization**: Interactive 3D view with rotation, zoom, and pan
        - **2D Layer View**: Examine each layer individually with slider control
        - **Preset Patterns**: Quick-start designs including checkerboard, core-shell, layered, and more
        
        #### 2. AI Design Assistant
        - **Optimal Pattern Generation**: Specify requirements and let AI generate optimized designs
        - **Pattern Analysis**: Get detailed mechanical analysis of your designs
        - **Property Prediction**: Predict Young's modulus, energy absorption, failure modes
        - **Design Comparison**: Compare multiple patterns side-by-side
        - **Creative Generation**: Generate novel patterns based on natural language descriptions
        
        #### 3. Chat Interface
        - Ask questions about metamaterial mechanics
        - Get design recommendations
        - Learn about specific material properties
        - Receive answers with scientific references
        
        ### Getting Started
        
        1. **Configure API Key**: Enter your Gemini API key in the sidebar
        2. **Select Grid Size**: Choose 2x2x2 or 4x4x4 grid
        3. **Load Preset**: Start with a preset pattern or create your own
        4. **Edit or Generate**: Use manual editing or AI generation
        5. **Visualize**: View in 3D or examine individual layers
        6. **Analyze**: Get AI-powered analysis and recommendations
        
        ### Design Guidelines
        
        **For High Stiffness:**
        - Use more stretching-dominated cells (red)
        - Place stiff cells in load paths
        - Consider vertical alignment for compression
        
        **For Energy Absorption:**
        - Use more bending-dominated cells (blue)
        - Distribute flexible cells throughout
        - Consider progressive collapse patterns
        
        **For Lightweight Structures:**
        - Balance between cell types
        - Use core-shell or sandwich patterns
        - Minimize stretching cells where not needed
        
        ### Applications
        
        - **Aerospace**: Lightweight structural components
        - **Automotive**: Crash absorption structures
        - **Biomedical**: Bone scaffolds, prosthetics
        - **Civil Engineering**: Seismic damping, bridge components
        - **Sports Equipment**: Protective gear, performance equipment
        
        ### Tips for Best Results
        
        1. Start with preset patterns and modify them
        2. Use AI analysis to understand pattern behavior
        3. Iterate between manual editing and AI suggestions
        4. Consider manufacturing constraints in your designs
        5. Validate designs with property predictions
        
        ### Scientific Background
        
        Hybrid architectured materials combine different unit cell topologies to achieve 
        superior mechanical properties. By strategically placing bending-dominated and 
        stretching-dominated cells, designers can create structures that are:
        
        - Stiffer than uniform materials
        - More energy-absorbing than monolithic structures
        - Lighter than solid components
        - Tailored for specific loading conditions
        
        ### References
        
        - Gibson & Ashby: "Cellular Solids: Structure and Properties"
        - Schaedler et al.: "Architected Materials: Theory and Practice"
        - Zheng et al.: "Ultralight, Ultrastiff Mechanical Metamaterials"
        - Fleck et al.: "Micro-Architectured Materials: Past, Present and Future"
        """)
        
        st.markdown("---")
        st.markdown("### Support")
        st.info("""
        For questions or issues:
        - Check the chat interface for technical questions
        - Review AI analysis for design guidance
        - Experiment with different patterns and configurations
        """)

if __name__ == "__main__":
    main()
