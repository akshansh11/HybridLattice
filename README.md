# HybridLattice: Agentic AI Model for Hybrid Architected Materials

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)

An advanced AI-powered web application for designing and analyzing hybrid lattice structures combining bending-dominated and stretching-dominated unit cells based on published research in metamaterials.

---

## Features

### Design Studio
- **Interactive 3D Visualization**: Real-time rendering of complete research-accurate lattice geometries
- **Manual Pattern Editor**: Layer-by-layer design interface with instant visual feedback
- **2D Layer View**: Cross-sectional analysis with interactive layer navigation
- **Preset Patterns**: Pre-configured designs including Checkerboard, Core-Shell, Layered, and Sandwich structures

### AI-Powered Design Assistant
- **Optimal Pattern Generation**: AI generates designs based on specific mechanical requirements
- **Pattern Analysis**: Comprehensive evaluation of stiffness, energy absorption, and failure modes
- **Property Prediction**: Estimates Young's modulus, Poisson's ratios, and mechanical performance
- **Scientific References**: AI-backed responses with relevant research citations

### Research-Accurate Unit Cells

**Bending-Dominated Cells** (Sky Blue):
- BCC (Body-Centered Cubic) - Diagonal strut architecture
- Rhombic Dodecahedron - 12 rhombic faces
- Kelvin Cell - Truncated octahedron, space-filling

**Stretching-Dominated Cells** (Red):
- Octet-Truss - Triangulated structure
- FCC (Face-Centered Cubic) - Face-centered connections
- Diamond Lattice - Tetrahedral network

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google Gemini API key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/akshansh11/HybridLattice.git
cd HybridLattice
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create requirements.txt** (if not included)
```text
streamlit>=1.28.0
numpy>=1.24.0
plotly>=5.17.0
google-generativeai>=0.3.0
```

---

## Usage

### Starting the Application

1. **Launch the Streamlit app**
```bash
streamlit run main.py
```

2. **Configure Gemini API**
   - Enter your Google Gemini API key in the sidebar
   - Get a free API key at: https://makersuite.google.com/app/apikey
   - Click "Configure API" to activate

3. **Design Your Lattice Structure**
   - Select grid size (2x2x2 or 4x4x4)
   - Choose unit cell types for bending and stretching regions
   - Load a preset pattern or design manually
   - Use the AI assistant for optimized designs

4. **Analyze and Visualize**
   - View your design in 3D or 2D layer view
   - Get AI-powered mechanical analysis
   - Review pattern statistics and predictions

---

## Application Structure

```
HybridLattice/
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # CC BY-NC-ND 4.0 License
└── assets/                     # Screenshots and images (optional)
```

---

## Configuration Options

### Grid Sizes
- **2x2x2**: Rapid prototyping and testing
- **4x4x4**: Detailed design and analysis

### Design Parameters
- **Loading Direction**: Z-axis compression, X-axis, Y-axis, or multi-directional
- **Stiffness Level**: Scale 1-10
- **Energy Absorption**: Percentage requirement (0-100%)
- **Weight Minimization**: Priority level (1-10)

### Unit Cell Selection
- Choose from 3 bending-dominated cell types
- Choose from 3 stretching-dominated cell types
- Real-time visualization updates

---

## Scientific Background

This application is based on peer-reviewed research in metamaterials and cellular solids:

### Key References
- **Gibson & Ashby**: Cellular Solids (1997) - Foundational work on foam mechanics
- **Deshpande et al.**: Foam Topology: Bending vs Stretching - Classification of deformation mechanisms
- **Zhang et al.**: Mechanical Performance of TPMS structures - Advanced lattice geometries
- **Ashby**: Materials Selection in Mechanical Design - Property optimization

### Deformation Mechanisms
- **Bending-dominated structures**: Higher compliance, greater energy absorption, suited for impact protection
- **Stretching-dominated structures**: Higher stiffness, superior strength-to-weight ratio, suited for load-bearing

---

## AI Assistant Capabilities

### Pattern Generation
The AI can generate optimal hybrid patterns based on:
- Loading conditions and directions
- Stiffness requirements
- Energy absorption targets
- Weight constraints

### Analysis Features
- Directional stiffness estimation (X, Y, Z axes)
- Failure mode prediction
- Application recommendations
- Design improvement suggestions

### Property Prediction
- Young's Modulus (Ex, Ey, Ez) in GPa
- Poisson's ratios
- Energy absorption capacity (J/cm³)
- Relative density percentage
- Failure stress (MPa)

---

## Example Workflows

### Workflow 1: Quick Design with Presets
1. Configure API key
2. Select grid size
3. Choose unit cell types
4. Load a preset pattern
5. Visualize in 3D
6. Get AI analysis

### Workflow 2: AI-Optimized Design
1. Configure API key
2. Go to "AI Assistant" tab
3. Select "Generate Optimal Pattern"
4. Set design requirements
5. Generate and review AI design
6. Visualize in Design Studio

### Workflow 3: Manual Custom Design
1. Configure API key
2. Load a preset as starting point
3. Use manual editor to modify layers
4. Toggle cells between bending/stretching
5. Analyze with AI
6. Iterate based on feedback

---

## Technical Details

### Geometry Implementation
All unit cell geometries are implemented based on published research:
- Complete strut networks with accurate connectivity
- Research-validated topologies
- Proper scaling and spacing
- Interactive 3D rendering with Plotly

### AI Integration
- Google Gemini 2.0 Flash model
- Scientific reference extraction
- Pattern parsing and validation
- Conversational interface

### Visualization
- Plotly 3D scatter plots for lattice structures
- Color-coded cell types (sky blue = bending, red = stretching)
- Interactive camera controls
- 2D layer-by-layer heatmaps

---

## Troubleshooting

### Common Issues

**Issue**: API key not working
- **Solution**: Verify key is correct at https://makersuite.google.com/app/apikey
- Check internet connection
- Ensure API quotas are not exceeded

**Issue**: Visualization not loading
- **Solution**: Refresh the page
- Check browser console for errors
- Try a smaller grid size (2x2x2)

**Issue**: AI not generating patterns
- **Solution**: Ensure API is configured
- Check that requirements are clearly specified
- Try simplifying the design constraints

---

## Contributing

This project is licensed under **CC BY-NC-ND 4.0**, which does not permit derivative works or commercial use. However, contributions are welcome in the following ways:

### Feedback and Bug Reports
1. Open an issue on GitHub
2. Describe the bug or suggestion clearly
3. Include screenshots if applicable
4. Provide steps to reproduce (for bugs)

### Feature Requests
- Suggest new unit cell types
- Propose additional analysis features
- Recommend UI improvements

**Note**: All contributions must respect the non-derivative clause of the license.

---

## License

**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**

### You are free to:
- **Share**: Copy and redistribute the material in any medium or format

### Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial**: You may not use the material for commercial purposes
- **NoDerivatives**: If you remix, transform, or build upon the material, you may not distribute the modified material

### Full License
The full license text is available at:
https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode

**Summary**: This software is free for educational, research, and non-commercial use. You cannot modify or create derivative works, and commercial use is prohibited.

---

## Citation

If you use this software in your research, please cite:

```
HybridLattice: Agentic AI Model for Hybrid Architected Materials
[Akshansh Mishra]
[2025]
Available at: https://github.com/akshansh11/HybridLattice
```

---

## Acknowledgments

This project builds upon decades of research in cellular solids and metamaterials. We acknowledge the foundational work of:
- Lorna J. Gibson and Michael F. Ashby
- Vikram S. Deshpande and Norman A. Fleck
- And numerous researchers in the field of architected materials

---

## Contact

For questions, suggestions, or collaboration inquiries:
- **GitHub Issues**: [Just mail me]
- **Email**: [akshanshmishra11@gmail.com]
- **Website**: [akshanshmishra.com]

---

## Version History

### Version 1.0.0 (Current)
- Initial release
- 6 unit cell types (3 bending, 3 stretching)
- AI-powered pattern generation and analysis
- Interactive 3D visualization
- Manual pattern editor
- Preset patterns

---

## Future Development

Planned features (subject to license restrictions):
- Additional unit cell geometries
- Export to CAD formats (STL, STEP)
- Advanced material property calculations
- Finite element analysis integration
- Multi-objective optimization

---

**Note**: This is research software provided as-is. Users are responsible for validating results for their specific applications.

---

Copyright (c) 2025 [Akshansh Mishra]. Licensed under CC BY-NC-ND 4.0.
