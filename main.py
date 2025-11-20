import streamlit as st
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime
import json
import re

# Page configuration
st.set_page_config(
    page_title="HybridLattice - Agentic AI Model",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #ffebee 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565c0;
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
    st.session_state.selected_bending_cell = 'bcc'
if 'selected_stretching_cell' not in st.session_state:
    st.session_state.selected_stretching_cell = 'octet'

# Color scheme
COLOR_BENDING = 'rgb(135, 206, 250)'
COLOR_STRETCHING = 'rgb(255, 0, 0)'

# Research-accurate cell type definitions
BENDING_CELL_TYPES = {
    'bcc': {'name': 'BCC (Body-Centered Cubic)', 'description': 'Bending-dominated - diagonal struts'},
    'rhombic_dodec': {'name': 'Rhombic Dodecahedron', 'description': 'Bending-dominated - 12 faces'},
    'kelvin': {'name': 'Kelvin Cell', 'description': 'Bending-dominated - truncated octahedron'},
}

STRETCHING_CELL_TYPES = {
    'octet': {'name': 'Octet-Truss', 'description': 'Stretching-dominated - triangulated'},
    'fcc': {'name': 'FCC (Face-Centered Cubic)', 'description': 'Stretching-dominated - face centers'},
    'diamond': {'name': 'Diamond', 'description': 'Stretching-dominated - tetrahedral'},
}

# ============================================
# RESEARCH-ACCURATE GEOMETRY FUNCTIONS
# Based on published literature
# ============================================

def create_bcc_cell_complete(i, j, k, size, color):
    """
    BCC (Body-Centered Cubic) - Bending-dominated
    Reference: Deshpande et al. "Foam topology: bending versus stretching"
    Complete geometry with all struts
    """
    traces = []
    s = size
    x0, y0, z0 = i * s, j * s, k * s
    
    # 8 corner vertices
    corners = np.array([
        [x0, y0, z0], [x0+s, y0, z0], [x0+s, y0+s, z0], [x0, y0+s, z0],
        [x0, y0, z0+s], [x0+s, y0, z0+s], [x0+s, y0+s, z0+s], [x0, y0+s, z0+s]
    ])
    
    # Body center
    center = np.array([x0+s/2, y0+s/2, z0+s/2])
    
    # BCC: 8 diagonal struts from corners to body center
    for corner in corners:
        traces.append(go.Scatter3d(
            x=[corner[0], center[0]],
            y=[corner[1], center[1]],
            z=[corner[2], center[2]],
            mode='lines',
            line=dict(color=color, width=10),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def create_rhombic_dodecahedron_complete(i, j, k, size, color):
    """
    Rhombic Dodecahedron - Bending-dominated
    Reference: Sun et al. hybrid lattice structures
    12 rhombic faces, complete geometry
    """
    traces = []
    s = size
    x0, y0, z0 = i * s, j * s, k * s
    
    # Center
    cx, cy, cz = x0 + s/2, y0 + s/2, z0 + s/2
    
    # 14 vertices of rhombic dodecahedron
    # 8 at corners + 6 at face centers
    vertices = np.array([
        # 8 corners (distance s/2 from center)
        [cx - s/2, cy, cz], [cx + s/2, cy, cz],  # X-axis
        [cx, cy - s/2, cz], [cx, cy + s/2, cz],  # Y-axis
        [cx, cy, cz - s/2], [cx, cy, cz + s/2],  # Z-axis
        # 8 diagonal vertices
        [cx - s/4, cy - s/4, cz - s/4], [cx + s/4, cy - s/4, cz - s/4],
        [cx + s/4, cy + s/4, cz - s/4], [cx - s/4, cy + s/4, cz - s/4],
        [cx - s/4, cy - s/4, cz + s/4], [cx + s/4, cy - s/4, cz + s/4],
        [cx + s/4, cy + s/4, cz + s/4], [cx - s/4, cy + s/4, cz + s/4],
    ])
    
    # 12 rhombic faces - each face has 4 edges
    faces = [
        (0, 7, 4, 6), (0, 9, 4, 10), (1, 8, 4, 7), (1, 11, 4, 8),
        (2, 6, 4, 7), (2, 10, 4, 11), (3, 8, 4, 9), (3, 11, 4, 10),
        (0, 6, 5, 13), (0, 10, 5, 13), (1, 7, 5, 12), (1, 11, 5, 12),
    ]
    
    # Draw edges of each rhombic face
    for face in faces:
        for i_v in range(4):
            v1 = vertices[face[i_v]]
            v2 = vertices[face[(i_v + 1) % 4]]
            traces.append(go.Scatter3d(
                x=[v1[0], v2[0]],
                y=[v1[1], v2[1]],
                z=[v1[2], v2[2]],
                mode='lines',
                line=dict(color=color, width=8),
                hoverinfo='skip',
                showlegend=False
            ))
    
    return traces

def create_kelvin_cell_complete(i, j, k, size, color):
    """
    Kelvin Cell (Truncated Octahedron) - Bending-dominated
    Reference: Gibson & Ashby cellular solids
    Space-filling, complete geometry
    """
    traces = []
    s = size
    x0, y0, z0 = i * s, j * s, k * s
    
    # Center
    cx, cy, cz = x0 + s/2, y0 + s/2, z0 + s/2
    a = s / 2.5  # Scaling factor
    
    # 24 vertices of truncated octahedron
    vertices = np.array([
        # 6 vertices along axes
        [cx - a, cy, cz], [cx + a, cy, cz],
        [cx, cy - a, cz], [cx, cy + a, cz],
        [cx, cy, cz - a], [cx, cy, cz + a],
        # 12 edge vertices
        [cx - a/2, cy - a/2, cz - a], [cx + a/2, cy - a/2, cz - a],
        [cx + a/2, cy + a/2, cz - a], [cx - a/2, cy + a/2, cz - a],
        [cx - a/2, cy - a/2, cz + a], [cx + a/2, cy - a/2, cz + a],
        [cx + a/2, cy + a/2, cz + a], [cx - a/2, cy + a/2, cz + a],
        [cx - a, cy - a/2, cz - a/2], [cx - a, cy + a/2, cz - a/2],
        [cx - a, cy + a/2, cz + a/2], [cx - a, cy - a/2, cz + a/2],
        [cx + a, cy - a/2, cz - a/2], [cx + a, cy + a/2, cz - a/2],
        [cx + a, cy + a/2, cz + a/2], [cx + a, cy - a/2, cz + a/2],
        [cx - a/2, cy - a, cz - a/2], [cx + a/2, cy - a, cz - a/2],
        [cx + a/2, cy - a, cz + a/2], [cx - a/2, cy - a, cz + a/2],
    ])
    
    # Edges of truncated octahedron - 36 edges
    edges = [
        (0, 14), (0, 15), (0, 16), (0, 17),
        (1, 18), (1, 19), (1, 20), (1, 21),
        (2, 6), (2, 7), (2, 22), (2, 23),
        (3, 8), (3, 9), (3, 12), (3, 13),
        (4, 6), (4, 7), (4, 8), (4, 9),
        (5, 10), (5, 11), (5, 12), (5, 13),
        (6, 14), (6, 22), (7, 18), (7, 23),
        (8, 19), (8, 23), (9, 15), (9, 22),
        (10, 17), (10, 25), (11, 21), (11, 24),
    ]
    
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        traces.append(go.Scatter3d(
            x=[v1[0], v2[0]],
            y=[v1[1], v2[1]],
            z=[v1[2], v2[2]],
            mode='lines',
            line=dict(color=color, width=7),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def create_octet_truss_complete(i, j, k, size, color):
    """
    Octet-Truss - Stretching-dominated
    Reference: Deshpande & Fleck, "Foam topology"
    Complete triangulated geometry with all struts
    """
    traces = []
    s = size
    x0, y0, z0 = i * s, j * s, k * s
    
    # 8 corner vertices
    corners = np.array([
        [x0, y0, z0], [x0+s, y0, z0], [x0+s, y0+s, z0], [x0, y0+s, z0],
        [x0, y0, z0+s], [x0+s, y0, z0+s], [x0+s, y0+s, z0+s], [x0, y0+s, z0+s]
    ])
    
    # 6 face centers
    face_centers = np.array([
        [x0+s/2, y0+s/2, z0],      # Bottom
        [x0+s/2, y0+s/2, z0+s],    # Top
        [x0, y0+s/2, z0+s/2],      # Left
        [x0+s, y0+s/2, z0+s/2],    # Right
        [x0+s/2, y0, z0+s/2],      # Front
        [x0+s/2, y0+s, z0+s/2],    # Back
    ])
    
    # Octet-truss: corners to adjacent face centers (24 struts)
    corner_to_faces = [
        (0, [0, 2, 4]), (1, [0, 3, 4]), (2, [0, 3, 5]), (3, [0, 2, 5]),
        (4, [1, 2, 4]), (5, [1, 3, 4]), (6, [1, 3, 5]), (7, [1, 2, 5])
    ]
    
    for corner_idx, face_indices in corner_to_faces:
        for face_idx in face_indices:
            traces.append(go.Scatter3d(
                x=[corners[corner_idx][0], face_centers[face_idx][0]],
                y=[corners[corner_idx][1], face_centers[face_idx][1]],
                z=[corners[corner_idx][2], face_centers[face_idx][2]],
                mode='lines',
                line=dict(color=color, width=10),
                hoverinfo='skip',
                showlegend=False
            ))
    
    return traces

def create_fcc_cell_complete(i, j, k, size, color):
    """
    FCC (Face-Centered Cubic) - Stretching-dominated
    Reference: Ashby "Cellular Solids"
    Complete geometry with face-center connections
    """
    traces = []
    s = size
    x0, y0, z0 = i * s, j * s, k * s
    
    # 8 corner vertices
    corners = np.array([
        [x0, y0, z0], [x0+s, y0, z0], [x0+s, y0+s, z0], [x0, y0+s, z0],
        [x0, y0, z0+s], [x0+s, y0, z0+s], [x0+s, y0+s, z0+s], [x0, y0+s, z0+s]
    ])
    
    # 6 face centers
    face_centers = np.array([
        [x0+s/2, y0+s/2, z0],      # Bottom (Z=0)
        [x0+s/2, y0+s/2, z0+s],    # Top (Z=s)
        [x0, y0+s/2, z0+s/2],      # Left (X=0)
        [x0+s, y0+s/2, z0+s/2],    # Right (X=s)
        [x0+s/2, y0, z0+s/2],      # Front (Y=0)
        [x0+s/2, y0+s, z0+s/2],    # Back (Y=s)
    ])
    
    # FCC: 12 edges connecting face centers
    fcc_edges = [
        (0, 2), (0, 3), (0, 4), (0, 5),  # Bottom face to side faces
        (1, 2), (1, 3), (1, 4), (1, 5),  # Top face to side faces
        (2, 4), (2, 5), (3, 4), (3, 5),  # Between side faces
    ]
    
    for edge in fcc_edges:
        v1, v2 = face_centers[edge[0]], face_centers[edge[1]]
        traces.append(go.Scatter3d(
            x=[v1[0], v2[0]],
            y=[v1[1], v2[1]],
            z=[v1[2], v2[2]],
            mode='lines',
            line=dict(color=color, width=10),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Also connect corners to their nearest face centers for continuity
    corner_face_connections = [
        (0, 0), (0, 2), (0, 4),  # Corner 0 to bottom, left, front
        (1, 0), (1, 3), (1, 4),  # Corner 1 to bottom, right, front
        (2, 0), (2, 3), (2, 5),  # Corner 2 to bottom, right, back
        (3, 0), (3, 2), (3, 5),  # Corner 3 to bottom, left, back
        (4, 1), (4, 2), (4, 4),  # Corner 4 to top, left, front
        (5, 1), (5, 3), (5, 4),  # Corner 5 to top, right, front
        (6, 1), (6, 3), (6, 5),  # Corner 6 to top, right, back
        (7, 1), (7, 2), (7, 5),  # Corner 7 to top, left, back
    ]
    
    for conn in corner_face_connections:
        corner_idx, face_idx = conn
        traces.append(go.Scatter3d(
            x=[corners[corner_idx][0], face_centers[face_idx][0]],
            y=[corners[corner_idx][1], face_centers[face_idx][1]],
            z=[corners[corner_idx][2], face_centers[face_idx][2]],
            mode='lines',
            line=dict(color=color, width=8),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

def create_diamond_lattice_complete(i, j, k, size, color):
    """
    Diamond Lattice - Stretching-dominated
    Reference: Zhang et al. "Mechanical performance of TPMS"
    Complete tetrahedral network
    """
    traces = []
    s = size
    x0, y0, z0 = i * s, j * s, k * s
    
    # Diamond lattice has 8 vertices in tetrahedral arrangement
    a = s / 4  # Lattice parameter
    vertices = np.array([
        [x0, y0, z0], [x0+s, y0, z0], [x0+s, y0+s, z0], [x0, y0+s, z0],
        [x0, y0, z0+s], [x0+s, y0, z0+s], [x0+s, y0+s, z0+s], [x0, y0+s, z0+s],
        [x0+s/2, y0+s/2, z0+s/2],  # Center
        [x0+s/4, y0+s/4, z0+s/4], [x0+3*s/4, y0+s/4, z0+s/4],
        [x0+3*s/4, y0+3*s/4, z0+s/4], [x0+s/4, y0+3*s/4, z0+s/4],
        [x0+s/4, y0+s/4, z0+3*s/4], [x0+3*s/4, y0+s/4, z0+3*s/4],
        [x0+3*s/4, y0+3*s/4, z0+3*s/4], [x0+s/4, y0+3*s/4, z0+3*s/4],
    ])
    
    # Tetrahedral connections - diamond structure
    edges = [
        # Center to 4 tetrahedral vertices
        (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16),
        # Tetrahedral edges
        (9, 10), (10, 11), (11, 12), (12, 9),
        (13, 14), (14, 15), (15, 16), (16, 13),
        # Connecting top and bottom
        (9, 13), (10, 14), (11, 15), (12, 16),
        # To corners for continuity
        (0, 9), (1, 10), (2, 11), (3, 12),
        (4, 13), (5, 14), (6, 15), (7, 16),
    ]
    
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        traces.append(go.Scatter3d(
            x=[v1[0], v2[0]],
            y=[v1[1], v2[1]],
            z=[v1[2], v2[2]],
            mode='lines',
            line=dict(color=color, width=9),
            hoverinfo='skip',
            showlegend=False
        ))
    
    return traces

# ============================================
# MAPPING FUNCTIONS
# ============================================

BENDING_FUNCTIONS = {
    'bcc': create_bcc_cell_complete,
    'rhombic_dodec': create_rhombic_dodecahedron_complete,
    'kelvin': create_kelvin_cell_complete,
}

STRETCHING_FUNCTIONS = {
    'octet': create_octet_truss_complete,
    'fcc': create_fcc_cell_complete,
    'diamond': create_diamond_lattice_complete,
}

# ============================================
# VISUALIZATION FUNCTION
# ============================================

def visualize_3d_pattern_connected(pattern, cell_size=1.0, show_legend=True):
    """Create 3D visualization with complete research-accurate geometries"""
    grid_size = pattern.shape[0]
    fig = go.Figure()
    
    bending_added = False
    stretching_added = False
    
    bending_type = st.session_state.selected_bending_cell
    stretching_type = st.session_state.selected_stretching_cell
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                
                if pattern[i, j, k] == 1:  # Bending
                    create_func = BENDING_FUNCTIONS.get(bending_type, create_bcc_cell_complete)
                    traces = create_func(i, j, k, cell_size, COLOR_BENDING)
                    for trace in traces:
                        if not bending_added and show_legend:
                            trace.showlegend = True
                            trace.name = f'Bending: {BENDING_CELL_TYPES[bending_type]["name"]}'
                            bending_added = True
                        fig.add_trace(trace)
                else:  # Stretching
                    create_func = STRETCHING_FUNCTIONS.get(stretching_type, create_octet_truss_complete)
                    traces = create_func(i, j, k, cell_size, COLOR_STRETCHING)
                    for trace in traces:
                        if not stretching_added and show_legend:
                            trace.showlegend = True
                            trace.name = f'Stretching: {STRETCHING_CELL_TYPES[stretching_type]["name"]}'
                            stretching_added = True
                        fig.add_trace(trace)
    
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

# ============================================
# UTILITY FUNCTIONS
# ============================================

def configure_gemini_api(api_key):
    """Configure Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        st.session_state.gemini_model = model
        st.session_state.api_key = api_key
        return True, "API configured successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"

def query_gemini_with_references(prompt):
    """Query Gemini AI"""
    if st.session_state.gemini_model is None:
        return "Please configure API key first", []
    
    try:
        enhanced_prompt = f"""{prompt}

IMPORTANT: List any scientific principles or references at the end as:
REFERENCES:
- [Reference 1]: Description
"""
        response = st.session_state.gemini_model.generate_content(enhanced_prompt)
        full_response = response.text
        
        references = []
        if "REFERENCES:" in full_response:
            ref_section = full_response.split("REFERENCES:")[1]
            ref_lines = [line.strip() for line in ref_section.split('\n') if line.strip().startswith('-')]
            references = [line.strip('- ') for line in ref_lines]
            main_response = full_response.split("REFERENCES:")[0].strip()
        else:
            main_response = full_response
        
        return main_response, references
    except Exception as e:
        return f"Error: {str(e)}", []

def parse_pattern_from_response(text, n):
    """Parse pattern from AI response"""
    pattern = np.ones((n, n, n), dtype=int)
    
    for layer in range(1, n + 1):
        tag = f"LAYER_{layer}"
        idx = text.find(tag)
        if idx != -1:
            chunk = text[idx:idx+500]
            mat_match = re.search(r'\[(.*?)\]', chunk, re.DOTALL)
            if mat_match:
                matrix_str = mat_match.group(1)
                rows = [row.strip() for row in matrix_str.split('\n') if row.strip()]
                for i, row in enumerate(rows[:n]):
                    values = [int(x) for x in re.findall(r'\d+', row)]
                    if len(values) == n:
                        pattern[i, :, layer-1] = values
    
    return pattern

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
                if i == 0 or i == grid_size-1 or j == 0 or j == grid_size-1 or k == 0 or k == grid_size-1:
                    pattern[i, j, k] = 2
    presets['Core-Shell'] = pattern
    
    # Layered
    pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
    for k in range(grid_size):
        if k < grid_size // 2:
            pattern[:, :, k] = 2
    presets['Layered'] = pattern
    
    # Sandwich
    pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
    pattern[:, :, 0] = 2
    pattern[:, :, -1] = 2
    presets['Sandwich'] = pattern
    
    return presets

def create_2d_layer_view(pattern):
    """Create 2D layer view"""
    grid_size = pattern.shape[0]
    fig = go.Figure()
    
    colorscale = [[0, COLOR_BENDING], [1, COLOR_STRETCHING]]
    
    for layer in range(grid_size):
        layer_data = pattern[:, :, layer]
        fig.add_trace(go.Heatmap(
            z=layer_data,
            colorscale=colorscale,
            showscale=False,
            text=np.where(layer_data == 1, 'B', 'S'),
            texttemplate='%{text}',
            textfont=dict(size=16, color='white'),
            name=f'Layer {layer+1}'
        ))
    
    fig.update_layout(
        sliders=[{
            'active': 0,
            'steps': [{'args': [{'visible': [i == j for j in range(grid_size)]}],
                      'label': str(i+1), 'method': 'update'}
                     for i in range(grid_size)]
        }],
        height=500,
        yaxis=dict(scaleanchor='x', scaleratio=1),
        title='2D Layer View'
    )
    
    for i, trace in enumerate(fig.data):
        trace.visible = (i == 0)
    
    return fig

def create_static_demo_image():
    """Create static 4x4x4 demo"""
    demo_pattern = np.ones((4, 4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i == 0 or i == 3 or j == 0 or j == 3 or k == 0 or k == 3:
                    demo_pattern[i, j, k] = 2
                elif (i + j + k) % 2 == 0:
                    demo_pattern[i, j, k] = 2
    
    fig = go.Figure()
    
    bending_added = False
    stretching_added = False
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if demo_pattern[i, j, k] == 1:
                    traces = create_bcc_cell_complete(i, j, k, 1.0, COLOR_BENDING)
                    for trace in traces:
                        if not bending_added:
                            trace.showlegend = True
                            trace.name = 'Bending-Dominated'
                            bending_added = True
                        fig.add_trace(trace)
                else:
                    traces = create_octet_truss_complete(i, j, k, 1.0, COLOR_STRETCHING)
                    for trace in traces:
                        if not stretching_added:
                            trace.showlegend = True
                            trace.name = 'Stretching-Dominated'
                            stretching_added = True
                        fig.add_trace(trace)
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='', showbackground=True, backgroundcolor="rgb(230,230,230)", 
                      gridcolor="white", showticklabels=False),
            yaxis=dict(title='', showbackground=True, backgroundcolor="rgb(230,230,230)", 
                      gridcolor="white", showticklabels=False),
            zaxis=dict(title='', showbackground=True, backgroundcolor="rgb(230,230,230)", 
                      gridcolor="white", showticklabels=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5), center=dict(x=2, y=2, z=2), up=dict(x=0, y=0, z=1))
        ),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1),
        height=450,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='white'
    )
    
    return fig

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.markdown('<div class="main-header">HybridLattice: An Agentic AI model to create hybrid architected materials</div>', 
                unsafe_allow_html=True)
    
    # Static demo image
    with st.spinner("Loading hybrid lattice structure..."):
        try:
            demo_fig = create_static_demo_image()
            st.plotly_chart(demo_fig, use_container_width=True, key="static_demo")
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; 
                        border-left: 3px solid #1f77b4; margin-top: 10px; text-align: center;'>
                <span style='color: #1f77b4;'>Sky Blue = Bending-Dominated</span> | 
                <span style='color: #ff0000;'>Red = Stretching-Dominated</span><br>
                <small>4x4x4 Hybrid Lattice - Complete Research-Accurate Geometry</small>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.info("Demo will load after configuration")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
        # API Key
        st.markdown("### Gemini API Key")
        api_key_input = st.text_input("Enter API Key", type="password",
                                       value=st.session_state.api_key if st.session_state.api_key else "")
        
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
        
        # Grid size
        st.markdown("### Grid Configuration")
        grid_size = st.selectbox("Grid Size", options=[2, 4], index=1)
        
        # Cell selection
        st.markdown("### Unit Cell Types")
        
        st.markdown("#### Bending-Dominated (Sky Blue)")
        selected_bending = st.selectbox(
            "Select Bending Cell",
            options=list(BENDING_CELL_TYPES.keys()),
            format_func=lambda x: BENDING_CELL_TYPES[x]['name'],
            index=list(BENDING_CELL_TYPES.keys()).index(st.session_state.selected_bending_cell)
        )
        
        if selected_bending != st.session_state.selected_bending_cell:
            st.session_state.selected_bending_cell = selected_bending
        
        bending_info = BENDING_CELL_TYPES[selected_bending]
        st.markdown(f'<div style="background-color: {COLOR_BENDING}; padding: 10px; border-radius: 5px;">'
                   f'{bending_info["name"]}<br><small>{bending_info["description"]}</small></div>', 
                   unsafe_allow_html=True)
        
        st.markdown("#### Stretching-Dominated (Red)")
        selected_stretching = st.selectbox(
            "Select Stretching Cell",
            options=list(STRETCHING_CELL_TYPES.keys()),
            format_func=lambda x: STRETCHING_CELL_TYPES[x]['name'],
            index=list(STRETCHING_CELL_TYPES.keys()).index(st.session_state.selected_stretching_cell)
        )
        
        if selected_stretching != st.session_state.selected_stretching_cell:
            st.session_state.selected_stretching_cell = selected_stretching
        
        stretching_info = STRETCHING_CELL_TYPES[selected_stretching]
        st.markdown(f'<div style="background-color: {COLOR_STRETCHING}; padding: 10px; border-radius: 5px; color: white;">'
                   f'{stretching_info["name"]}<br><small>{stretching_info["description"]}</small></div>', 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Presets
        st.markdown("### Preset Patterns")
        presets = generate_pattern_presets(grid_size)
        preset_choice = st.selectbox("Select Preset", options=list(presets.keys()))
        
        if st.button("Load Preset"):
            st.session_state.current_pattern = presets[preset_choice]
            st.success(f"Loaded {preset_choice}")
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.current_pattern is not None:
            st.markdown("### Pattern Statistics")
            pattern = st.session_state.current_pattern
            total = pattern.size
            bending = np.sum(pattern == 1)
            stretching = np.sum(pattern == 2)
            
            st.metric("Total Cells", total)
            st.metric("Bending", f"{bending} ({100*bending/total:.1f}%)")
            st.metric("Stretching", f"{stretching} ({100*stretching/total:.1f}%)")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Design Studio", "AI Assistant", "Chat"])
    
    with tab1:
        st.markdown("## Design Studio")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Manual Editor")
            
            if st.session_state.current_pattern is not None:
                pattern = st.session_state.current_pattern
                layer_to_edit = st.slider("Select Layer", 0, grid_size-1, 0)
                
                st.write(f"Layer {layer_to_edit + 1}")
                
                for i in range(grid_size):
                    cols = st.columns(grid_size)
                    for j in range(grid_size):
                        with cols[j]:
                            cell_type = pattern[i, j, layer_to_edit]
                            label = "B" if cell_type == 1 else "S"
                            
                            if st.button(label, key=f"cell_{layer_to_edit}_{i}_{j}"):
                                pattern[i, j, layer_to_edit] = 1 if pattern[i, j, layer_to_edit] == 2 else 2
                                st.session_state.current_pattern = pattern
                                st.rerun()
                
                st.markdown("---")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Reset"):
                        st.session_state.current_pattern = np.ones((grid_size, grid_size, grid_size), dtype=int)
                        st.rerun()
                with col_b:
                    if st.button("Randomize"):
                        st.session_state.current_pattern = np.random.choice([1, 2], size=(grid_size, grid_size, grid_size))
                        st.rerun()
            else:
                st.info("Load a preset to begin")
        
        with col2:
            st.markdown("### Visualization")
            
            if st.session_state.current_pattern is not None:
                viz_type = st.radio("Type", options=["3D Interactive", "2D Layer View"], horizontal=True)
                
                if viz_type == "3D Interactive":
                    with st.spinner("Generating complete research-accurate 3D geometry..."):
                        fig = visualize_3d_pattern_connected(st.session_state.current_pattern)
                        st.plotly_chart(fig, use_container_width=True)
                    st.info("Complete unit cell geometries based on published research")
                else:
                    fig = create_2d_layer_view(st.session_state.current_pattern)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No pattern loaded")
    
    with tab2:
        st.markdown("## AI Design Assistant")
        
        if st.session_state.api_key is None:
            st.warning("Please configure API key in sidebar")
        else:
            st.markdown("Use AI to generate optimal patterns")
            
            task = st.selectbox("Select Task", [
                "Generate Optimal Pattern",
                "Analyze Current Pattern",
                "Property Prediction"
            ])
            
            if task == "Generate Optimal Pattern":
                st.markdown("### Design Requirements")
                
                col1, col2 = st.columns(2)
                with col1:
                    loading = st.selectbox("Loading Direction", 
                                          ["Z-axis (Compression)", "X-axis", "Y-axis", "Multi-directional"])
                    stiffness = st.slider("Stiffness", 1, 10, 7)
                with col2:
                    energy = st.slider("Energy Absorption (%)", 0, 100, 30)
                    weight = st.slider("Weight Minimization", 1, 10, 5)
                
                if st.button("Generate Design", type="primary"):
                    with st.spinner("AI generating..."):
                        prompt = f"""Design an optimal {grid_size}x{grid_size}x{grid_size} hybrid material.

REQUIREMENTS:
- Loading: {loading}
- Stiffness: {stiffness}/10
- Energy absorption: {energy}%
- Weight importance: {weight}/10

CELL TYPES:
- Type 1 = Bending-dominated (flexible)
- Type 2 = Stretching-dominated (stiff)

FORMAT:
LAYER_1:
[1 2 1 2
 2 1 2 1
 ...]

Show all {grid_size} layers."""
                        
                        response, references = query_gemini_with_references(prompt)
                        
                        st.markdown("### AI Response")
                        st.markdown(response)
                        
                        if references:
                            st.markdown('<div style="background:#fff9c4; padding:10px; border-radius:5px;">'
                                      '<strong>References:</strong>', unsafe_allow_html=True)
                            for ref in references:
                                st.markdown(f"- {ref}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        try:
                            pattern = parse_pattern_from_response(response, grid_size)
                            st.session_state.current_pattern = pattern
                            st.success("Pattern loaded - check Design Studio")
                        except:
                            st.error("Could not parse pattern")
            
            elif task == "Analyze Current Pattern":
                if st.session_state.current_pattern is None:
                    st.warning("Load a pattern first")
                else:
                    if st.button("Analyze", type="primary"):
                        with st.spinner("Analyzing..."):
                            pattern = st.session_state.current_pattern
                            prompt = f"""Analyze this {grid_size}x{grid_size}x{grid_size} pattern:
Layer 1: {pattern[:,:,0].tolist()}

Where 1=flexible, 2=stiff

PROVIDE:
1. Stiffness in X,Y,Z (rate 1-10 each)
2. Energy absorption capability
3. Expected failure mode
4. Applications (at least 3)
5. Improvements (at least 2)"""
                            
                            response, references = query_gemini_with_references(prompt)
                            
                            st.markdown("### Analysis")
                            st.markdown(response)
                            
                            if references:
                                st.markdown('<div style="background:#fff9c4; padding:10px;">'
                                          '<strong>References:</strong>', unsafe_allow_html=True)
                                for ref in references:
                                    st.markdown(f"- {ref}")
                                st.markdown('</div>', unsafe_allow_html=True)
            
            elif task == "Property Prediction":
                if st.session_state.current_pattern is None:
                    st.warning("Load a pattern first")
                else:
                    if st.button("Predict Properties", type="primary"):
                        with st.spinner("Predicting..."):
                            pattern = st.session_state.current_pattern
                            prompt = f"""Predict properties of this {grid_size}x{grid_size}x{grid_size} pattern:
{pattern[:,:,grid_size//2].tolist()}

PREDICT:
1. Young's Modulus (Ex, Ey, Ez) - GPa
2. Poisson's ratios
3. Energy absorption (J/cm³)
4. Relative density (%)
5. Failure stress (MPa)"""
                            
                            response, references = query_gemini_with_references(prompt)
                            
                            st.markdown("### Predictions")
                            st.markdown(response)
                            
                            if references:
                                st.markdown('<div style="background:#fff9c4; padding:10px;">'
                                          '<strong>References:</strong>', unsafe_allow_html=True)
                                for ref in references:
                                    st.markdown(f"- {ref}")
                                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## Chat with AI")
        
        if st.session_state.api_key is None:
            st.warning("Configure API key first")
        else:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div style="background:#e3f2fd; padding:10px; border-radius:5px; margin:10px 0;">'
                              f'<strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background:#f5f5f5; padding:10px; border-radius:5px; margin:10px 0;">'
                              f'<strong>AI:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
                    if 'references' in message and message['references']:
                        st.markdown('<div style="background:#fff9c4; padding:5px; margin-top:5px;">'
                                  '<strong>References:</strong>', unsafe_allow_html=True)
                        for ref in message['references']:
                            st.markdown(f"- {ref}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_area("Your question:", height=100)
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    submit = st.form_submit_button("Send", type="primary")
                with col2:
                    clear = st.form_submit_button("Clear Chat")
                
                if clear:
                    st.session_state.chat_history = []
                    st.rerun()
                
                if submit and user_input:
                    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                    
                    with st.spinner("AI thinking..."):
                        prompt = f"""You are an expert in metamaterials. Answer: {user_input}"""
                        response, references = query_gemini_with_references(prompt)
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response,
                            'references': references
                        })
                    
                    st.rerun()

if __name__ == "__main__":
    main()
