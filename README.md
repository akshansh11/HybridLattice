# HybridLattice: An Agentic AI Model for designing Hybrid Architected Materials

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)
![Research](https://img.shields.io/badge/Research-Validated-green.svg)

An advanced AI-powered web application for designing and analyzing hybrid lattice structures combining bending-dominated and stretching-dominated unit cells based on **peer-reviewed research** in metamaterials and cellular solids.

> **NEW**: Research-validated compatibility checking ensures only scientifically proven cell combinations are used.

---

## Features

### Design Studio
- **Interactive 3D Visualization**: Real-time rendering of complete research-accurate lattice geometries
- **Research-Based Compatibility Checking**: Automatic validation of cell pairings based on published literature
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
- **BCC (Body-Centered Cubic)** - Z=8 - Diagonal strut architecture, excellent energy absorption
- **Rhombic Dodecahedron** - Z=12 - 12 rhombic faces, good bending resistance
- **Kelvin Cell** - Z=14 - Truncated octahedron, space-filling (*limited hybrid compatibility*)

**Stretching-Dominated Cells** (Red):
- **Octet-Truss** - Z=12 - Triangulated structure, highest stiffness
- **FCC (Face-Centered Cubic)** - Z=12 - Face-centered connections
- **Diamond Lattice** - Z=4 - Tetrahedral network (*limited hybrid compatibility*)

> **Note**: Z values indicate nodal connectivity - critical for proper hybrid lattice integration

---

## Research-Validated Cell Compatibility

### ✓ Compatible Pairings (Recommended)

#### 1. **BCC + Octet-Truss** ✓✓✓ (Highly Recommended)
- **Connectivity**: Z=8 → Z=12 (hierarchical distribution)
- **Mechanism**: Continuous distribution at shared boundary nodes
- **Research**: Bian et al. (2023), *ACS Applied Materials & Interfaces*, 15:15928-15937
- **Properties**: Balanced stiffness and energy absorption, excellent for impact resistance
- **Applications**: Aerospace structures, protective equipment, energy-absorbing components

#### 2. **BCC + FCC** ✓✓ (Recommended)
- **Connectivity**: Z=8 → Z=12 (hierarchical arrangement)
- **Mechanism**: Precipitation-inspired strengthening
- **Research**: Bian et al. (2023), *ACS Applied Materials & Interfaces*
- **Properties**: Enhanced toughness, synergistic phase interactions
- **Applications**: Structural frames, multi-directional loading components

#### 3. **Rhombic Dodecahedron + Octet-Truss** ✓✓✓ (Perfect Match)
- **Connectivity**: Z=12 → Z=12 (direct nodal match)
- **Mechanism**: Perfect connectivity with complementary deformation modes
- **Research**: Yu et al. (2023), *Vibroengineering Procedia*, 50:206-212
- **Properties**: Enhanced Young's modulus and stress magnitude, pattern-dependent optimization
- **Applications**: High-precision components, validated structural designs

### ✗ Incompatible Pairings (Not Recommended)

- **Kelvin Cell + Any**: Z=14 connectivity mismatch, no research validation
- **Diamond + Most Cells**: Z=4 too low for proper hybrid integration

> **Warning**: Using incompatible pairings may result in disconnected struts at cell boundaries and unpredictable mechanical behavior.

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
streamlit run hybrid_lattice_corrected.py
```

2. **Configure Gemini API**
   - Enter your Google Gemini API key in the sidebar
   - Get a free API key at: https://makersuite.google.com/app/apikey
   - Click "Configure API" to activate

3. **Select Compatible Cell Types**
   - Choose a bending-dominated cell (BCC or Rhombic Dodecahedron recommended)
   - The app will automatically filter compatible stretching cells
   - Look for the **✓ [COMPATIBLE]** green status indicator
   - Review the connection mechanism and research references

4. **Design Your Lattice Structure**
   - Select grid size (2x2x2 or 4x4x4)
   - Load a preset pattern or design manually
   - Use the AI assistant for optimized designs
   - View compatibility status at all times

5. **Analyze and Visualize**
   - View your design in 3D or 2D layer view
   - Get AI-powered mechanical analysis
   - Review pattern statistics and predictions
   - Export or iterate on your design

---

## Application Structure

```
HybridLattice/
├── hybrid_lattice_corrected.py    # Main application file (research-validated)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # CC BY-NC-ND 4.0 License
├── RESEARCH_CORRECTIONS.md        # Detailed research documentation
├── QUICK_REFERENCE.md             # Quick compatibility guide
└── assets/                        # Screenshots and images (optional)
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
- Automatically filtered compatible stretching cells
- Real-time compatibility validation
- Research-backed mechanism explanations

---

## Scientific Background

This application is based on **peer-reviewed research** in metamaterials and cellular solids:

### Primary Research References

#### 1. **Hierarchical Multiphase Lattices**
**Bian, Y., Wang, R., Yang, F., et al. (2023)**
- *Mechanical Properties of Internally Hierarchical Multiphase Lattices Inspired by Precipitation Strengthening Mechanisms*
- **Journal**: ACS Applied Materials & Interfaces, 15, 15928-15937
- **DOI**: 10.1021/acsami.2c20063
- **Key Finding**: Tri-phase lattices achieve balanced properties; introducing weaker phases can enhance stiffness (counter-intuitive)

#### 2. **Heterogeneous Lattice Structures**
**Yu, G., Miao, C., Wu, H., & Liang, J. (2023)**
- *Mechanical performance of heterogeneous lattice structure*
- **Journal**: Vibroengineering Procedia, 50, 206-212
- **DOI**: 10.21595/vp.2023.23454
- **Key Finding**: Rhombic dodecahedron (bending) + octet-truss (stretching) enhance Young's modulus and stress magnitude

#### 3. **Variable Density Composite Lattices**
**Zhang, M., Zhao, C., Li, G., & Luo, K. (2023)**
- *Mechanical properties of the composite lattice structure with variable density and multi-configuration*
- **Journal**: Composite Structures, 304, 116405
- **DOI**: 10.1016/j.compstruct.2022.116405
- **Key Finding**: Material redistribution and multi-configuration enhance bearing and impact resistance

### Foundational References
- **Gibson & Ashby**: Cellular Solids (1997) - Foam mechanics fundamentals
- **Deshpande et al.**: Foam Topology: Bending vs Stretching - Deformation classification
- **Ashby**: Materials Selection in Mechanical Design - Property optimization

### Deformation Mechanisms
- **Bending-dominated structures**: Higher compliance, greater energy absorption, suited for impact protection
- **Stretching-dominated structures**: Higher stiffness, superior strength-to-weight ratio, suited for load-bearing
- **Hybrid structures**: Combine advantages through proper nodal connectivity

---

## Nodal Connectivity Theory

### Why Z-Values Matter

**Nodal Connectivity (Z)** = Number of struts meeting at each node

#### Direct Match (Z=12 to Z=12):
```
Rhombic Dodecahedron (Z=12) ←→ Octet-Truss (Z=12)
              ↓                           ↓
    All nodes connect directly - PERFECT INTEGRATION
```

#### Hierarchical Distribution (Z=8 to Z=12):
```
BCC (Z=8) ←→ Octet-Truss (Z=12)
    ↓                 ↓
Continuous distribution at boundaries enables proper connectivity
```

#### Mismatch (Z=14 to Z=12):
```
Kelvin (Z=14) ←→ Octet-Truss (Z=12)
      ↓                    ↓
  DISCONNECTED - Cannot form proper structural integration
```

> **Critical**: Only use validated Z-value pairings to ensure structural integrity

---

## AI Assistant Capabilities

### Pattern Generation
The AI can generate optimal hybrid patterns based on:
- Loading conditions and directions
- Stiffness requirements
- Energy absorption targets
- Weight constraints
- **Cell compatibility constraints** (new)

### Analysis Features
- Directional stiffness estimation (X, Y, Z axes)
- Failure mode prediction
- Application recommendations
- Design improvement suggestions
- Compatibility assessment with research backing

### Property Prediction
- Young's Modulus (Ex, Ey, Ez) in GPa
- Poisson's ratios
- Energy absorption capacity (J/cm³)
- Relative density percentage
- Failure stress (MPa)
- Nodal connectivity validation

---

## Example Workflows

### Workflow 1: Research-Validated Quick Design
1. Configure API key
2. Select BCC as bending cell
3. **Observe automatic filtering** to Octet and FCC only
4. **Check green ✓ [COMPATIBLE] status**
5. Load a preset pattern
6. Visualize in 3D
7. Get AI analysis with research references

### Workflow 2: AI-Optimized Design with Compatibility
1. Configure API key
2. Select compatible cell pair (e.g., BCC + Octet)
3. Go to "AI Assistant" tab
4. Select "Generate Optimal Pattern"
5. Set design requirements
6. **AI generates pattern respecting cell compatibility**
7. Review research references provided
8. Visualize and analyze

### Workflow 3: Exploring Cell Combinations
1. Configure API key
2. Try BCC + Octet (see green ✓ status)
3. Try BCC + FCC (see green ✓ status)
4. Try Rhombic Dodec + Octet (see green ✓ - perfect match)
5. Try Kelvin + Any (see red ✗ warning)
6. **Learn about connectivity mechanisms**
7. Choose best for your application

### Workflow 4: Manual Design with Validation
1. Configure API key
2. Select Rhombic Dodecahedron + Octet (Z=12 match)
3. Load Core-Shell preset
4. Modify using manual editor
5. **Check compatibility status remains green**
6. Analyze with AI
7. Iterate based on feedback

---

## Technical Details

### Geometry Implementation
All unit cell geometries are implemented based on published research:
- Complete strut networks with accurate connectivity
- Research-validated topologies
- Proper Z-value representation
- Correct nodal positions for hybrid integration
- Interactive 3D rendering with Plotly

### Compatibility Validation System
```python
COMPATIBILITY_MATRIX = {
    'bcc': {
        'compatible': ['octet', 'fcc'],
        'reason': 'Hierarchical continuous distribution',
        'references': ['Bian et al. 2023', 'Zhang et al. 2023'],
        'mechanism': 'Z=8 to Z=12 via boundary nodes'
    },
    'rhombic_dodec': {
        'compatible': ['octet'],
        'reason': 'Perfect nodal connectivity match',
        'references': ['Yu et al. 2023'],
        'mechanism': 'Z=12 to Z=12 direct match'
    },
    # ...
}
```

### AI Integration
- Google Gemini 2.0 Flash model
- Scientific reference extraction
- Compatibility-aware pattern generation
- Research-backed recommendations
- Conversational interface with citations

### Visualization
- Plotly 3D scatter plots for lattice structures
- Color-coded cell types (sky blue = bending, red = stretching)
- Compatibility status overlays (green/red warnings)
- Interactive camera controls
- 2D layer-by-layer heatmaps
- Cell connectivity visualization

---

## Compatibility Status Indicators

### ✓ [COMPATIBLE] - Green Box
```
Cells can connect properly at boundaries
Research-validated pairing
Proper structural integration ensured
Full mechanism explanation provided
```

### ✗ [INCOMPATIBLE] - Red Box
```
WARNING: These cells cannot form proper connections
No research validation available
May show disconnected struts
Choose alternative pairing from sidebar
```

---

## Troubleshooting

### Common Issues

**Issue**: "No compatible stretching cells available"
- **Cause**: Selected Kelvin cell (Z=14) which has no compatible pairings
- **Solution**: Switch to BCC (Z=8) or Rhombic Dodecahedron (Z=12)

**Issue**: Red ✗ [INCOMPATIBLE] warning appears
- **Cause**: Selected pairing not validated by research
- **Solution**: Choose from recommended pairings:
  - BCC + Octet ✓✓✓
  - BCC + FCC ✓✓
  - Rhombic Dodecahedron + Octet ✓✓✓

**Issue**: API key not working
- **Solution**: Verify key at https://makersuite.google.com/app/apikey
- Check internet connection
- Ensure API quotas are not exceeded

**Issue**: Visualization not loading
- **Solution**: Refresh the page
- Check browser console for errors
- Try a smaller grid size (2x2x2)
- Ensure compatible cells are selected

**Issue**: AI not generating patterns
- **Solution**: Ensure API is configured
- Verify cells are compatible (green status)
- Check that requirements are clearly specified
- Try simplifying the design constraints

**Issue**: Disconnected struts visible in 3D
- **Cause**: Incompatible cell pairing selected
- **Solution**: This indicates structural problems - select compatible pairing immediately

---

## Best Practices

### For Maximum Success:

1. **Always verify green ✓ [COMPATIBLE] status** before designing
2. **Start with BCC + Octet** - most research support
3. **Read mechanism explanations** to understand connections
4. **Use Rhombic Dodec + Octet** for highest confidence (Z=12 match)
5. **Avoid Kelvin and Diamond cells** for hybrid structures
6. **Check research references** provided in compatibility box
7. **Let AI suggest patterns** based on requirements
8. **Iterate designs** using analysis feedback

---

## Contributing

This project is licensed under **CC BY-NC-ND 4.0**, which does not permit derivative works or commercial use. However, contributions are welcome in the following ways:

### Feedback and Bug Reports
1. Open an issue on GitHub
2. Describe the bug or suggestion clearly
3. Include compatibility status screenshots if applicable
4. Provide steps to reproduce (for bugs)

### Feature Requests
- Suggest new **validated** unit cell types with research papers
- Propose additional analysis features
- Recommend UI improvements
- Share research papers on new cell combinations

### Research Contributions
- Submit papers validating new cell pairings
- Provide experimental data supporting compatibility
- Share mechanical testing results
- Suggest additional compatibility mechanisms

**Note**: All contributions must respect the non-derivative clause of the license and be backed by peer-reviewed research.

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

```bibtex
@software{hybridlattice2025,
  title = {HybridLattice: Agentic AI Model for Hybrid Architected Materials},
  author = {Mishra, Akshansh},
  year = {2025},
  url = {https://github.com/akshansh11/HybridLattice},
  note = {Research-validated cell compatibility based on Bian et al. (2023), 
          Yu et al. (2023), and Zhang et al. (2023)}
}
```

### Also cite the foundational research:

```bibtex
@article{bian2023hierarchical,
  title = {Mechanical Properties of Internally Hierarchical Multiphase Lattices 
           Inspired by Precipitation Strengthening Mechanisms},
  author = {Bian, Yijie and Wang, Ruicheng and Yang, Fan and others},
  journal = {ACS Applied Materials \& Interfaces},
  volume = {15},
  pages = {15928--15937},
  year = {2023},
  doi = {10.1021/acsami.2c20063}
}

@article{yu2023heterogeneous,
  title = {Mechanical performance of heterogeneous lattice structure},
  author = {Yu, Guoji and Miao, Cheng and Wu, Hailing and Liang, Jiayi},
  journal = {Vibroengineering Procedia},
  volume = {50},
  pages = {206--212},
  year = {2023},
  doi = {10.21595/vp.2023.23454}
}

@article{zhang2023composite,
  title = {Mechanical properties of the composite lattice structure with 
           variable density and multi-configuration},
  author = {Zhang, Meng and Zhao, Cun and Li, Guoxi and Luo, Kai},
  journal = {Composite Structures},
  volume = {304},
  pages = {116405},
  year = {2023},
  doi = {10.1016/j.compstruct.2022.116405}
}
```

---

## Acknowledgments

This project builds upon decades of research in cellular solids and metamaterials. We acknowledge:

### Foundational Researchers
- Lorna J. Gibson and Michael F. Ashby - Cellular solids theory
- Vikram S. Deshpande and Norman A. Fleck - Foam topology classification

### Recent Research Contributions (2023)
- **Bian et al.** - Hierarchical multiphase lattice mechanisms
- **Yu et al.** - Heterogeneous lattice validation
- **Zhang et al.** - Variable density multi-configuration design

### Research Community
- All researchers advancing the field of architected materials
- Peer reviewers ensuring scientific rigor
- Open-access publishers making research available

---

## Contact

For questions, suggestions, or collaboration inquiries:
- **GitHub Issues**: [Create an issue](https://github.com/akshansh11/HybridLattice/issues)
- **Email**: akshanshmishra11@gmail.com
- **Website**: akshanshmishra.com

For research-related questions about cell compatibility:
- Refer to **RESEARCH_CORRECTIONS.md** for detailed explanations
- Refer to **QUICK_REFERENCE.md** for quick compatibility guide
- Check research paper references in the app

---

## Version History

### Version 2.0.0 (Current) - Research-Validated Release
- **NEW**: Research-based compatibility validation system
- **NEW**: Automatic cell filtering based on validated pairings
- **NEW**: Real-time compatibility status indicators
- **NEW**: Connection mechanism explanations
- **NEW**: Full research citations integrated
- **IMPROVED**: AI assistant respects compatibility constraints
- **IMPROVED**: Enhanced visualization with compatibility warnings
- **ADDED**: Detailed research documentation
- **ADDED**: Quick reference guide for cell pairings
- 6 unit cell types (3 bending, 3 stretching)
- Interactive 3D visualization
- Manual pattern editor
- Preset patterns

### Version 1.0.0
- Initial release
- Basic cell combinations without validation
- AI-powered pattern generation
- 3D visualization

---

## Future Development

Planned features (subject to license restrictions and research validation):

### Validated Additions
- Additional unit cell geometries with research backing
- Advanced material property calculations based on published models
- Multi-objective optimization with compatibility constraints

### Pending Research Validation
- Export to CAD formats (STL, STEP) with proper connectivity
- Finite element analysis integration
- Tri-phase lattice implementation (BCC + Octet + FCC)
- Variable density optimization (Zhang et al. 2023 algorithms)

### Research Needed
- New cell combinations requiring experimental validation
- Dynamic loading simulations
- Manufacturing constraint optimization

> All new features will require peer-reviewed research validation before implementation

---

## Important Notes

### Research Validation
✓ This software implements **only peer-reviewed, scientifically validated** cell combinations
✓ All compatibility rules are backed by published research from 2023
✓ Incompatible combinations are clearly flagged with warnings

### User Responsibility
- Users are responsible for validating results for their specific applications
- This is research software provided as-is for educational purposes
- Manufacturing feasibility should be verified independently
- Mechanical testing is recommended for critical applications

### Limitations
- Limited to validated cell pairings only (intentional safety feature)
- AI suggestions constrained by research-backed compatibility
- Some cell types have limited hybrid utility (Kelvin, Diamond)
- Requires internet connection for AI features

---

## Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Get Gemini API key
- [ ] Run `streamlit run hybrid_lattice_corrected.py`
- [ ] Configure API in sidebar
- [ ] Select **BCC** as bending cell
- [ ] Select **Octet** as stretching cell
- [ ] Verify **✓ [COMPATIBLE]** green status
- [ ] Load a preset pattern
- [ ] Explore 3D visualization
- [ ] Try AI analysis
- [ ] Read **QUICK_REFERENCE.md** for details

---

**Note**: This software prioritizes scientific accuracy over flexibility. Only research-validated combinations are permitted to ensure reliable mechanical performance predictions.

---

Copyright (c) 2025 Akshansh Mishra. Licensed under CC BY-NC-ND 4.0.

**Research-Validated Release** | **Version 2.0.0** | **Last Updated**: November 2025
