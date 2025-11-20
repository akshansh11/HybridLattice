# Quick Reference: Compatible Cell Pairings

## ✓ RESEARCH-VALIDATED COMPATIBLE COMBINATIONS

### 1. BCC + Octet ✓✓✓ (HIGHLY RECOMMENDED)
```
Bending: BCC (Z=8)  →  Stretching: Octet-Truss (Z=12)
```
**Why Compatible:**
- Hierarchical continuous distribution at shared nodes
- Most extensively studied combination in literature
- Excellent balance of stiffness and energy absorption

**Research Support:**
- Bian et al. (2023): ACS Applied Materials & Interfaces
- Zhang et al. (2023): Composite Structures 304
- Multiple papers on hybrid BCC-Octet structures

**Properties:**
- ✓ High stiffness from Octet
- ✓ High energy absorption from BCC
- ✓ Balanced mechanical performance
- ✓ Good manufacturability

**Use Cases:**
- Aerospace structures requiring impact resistance
- Energy-absorbing protective equipment
- Load-bearing lightweight components

---

### 2. BCC + FCC ✓✓ (RECOMMENDED)
```
Bending: BCC (Z=8)  →  Stretching: FCC (Z=12)
```
**Why Compatible:**
- Precipitation-inspired strengthening mechanism
- Hierarchical arrangement enables proper connectivity
- Synergistic interaction between phases

**Research Support:**
- Bian et al. (2023): Hierarchical Multiphase Lattices
- FCC-BCC hybrid structures validated

**Properties:**
- ✓ Enhanced stiffness compared to pure BCC
- ✓ Good toughness and ductility
- ✓ Precipitation-like strengthening effects

**Use Cases:**
- Structural components requiring toughness
- Multi-directional loading applications
- Lightweight frames and supports

---

### 3. Rhombic Dodecahedron + Octet ✓✓✓ (PERFECT MATCH)
```
Bending: Rhombic Dodecahedron (Z=12)  →  Stretching: Octet-Truss (Z=12)
```
**Why Compatible:**
- PERFECT nodal connectivity match (Z=12 to Z=12)
- Complementary deformation modes (bending + stretching)
- Direct structural integration at boundaries

**Research Support:**
- Yu et al. (2023): Vibroengineering Procedia 50:206-212
- Experimentally validated through compression tests
- SLM fabrication proven successful

**Properties:**
- ✓ Highest confidence in connectivity
- ✓ Enhanced Young's modulus
- ✓ Improved stress magnitude
- ✓ Pattern-dependent mechanical optimization

**Use Cases:**
- High-precision applications
- Critical structural components
- Applications requiring validated performance

---

## ✗ INCOMPATIBLE COMBINATIONS (NOT RECOMMENDED)

### Kelvin Cell + Any Stretching Cell ✗
```
Bending: Kelvin (Z=14)  →  Stretching: Any (Z=4, Z=12)
```
**Why Incompatible:**
- Z=14 connectivity cannot match Z=12 or Z=4
- No research validation for hybrid structures
- Disconnected struts at cell boundaries

**Warning:**
- ⚠ Do not use for functional structures
- ⚠ Visualization only - not manufacturable
- ⚠ No mechanical performance data available

---

### Diamond + Most Cells ✗
```
Stretching: Diamond (Z=4)  →  Bending: Most (Z=8, Z=12, Z=14)
```
**Why Limited:**
- Very low connectivity (Z=4)
- Limited research on hybrid configurations
- Difficult to achieve proper connections

**Note:**
- Some specialized configurations may work
- Requires specific hierarchical arrangements
- Not generally recommended without validation

---

## CONNECTIVITY (Z) REFERENCE TABLE

| Cell Type | Z Value | Deformation Mode | Hybrid Use |
|-----------|---------|------------------|------------|
| BCC | 8 | Bending | ✓✓✓ Excellent |
| FCC | 12 | Stretching | ✓✓ Good |
| Octet-Truss | 12 | Stretching | ✓✓✓ Excellent |
| Rhombic Dodec | 12 | Bending | ✓✓ Good |
| Kelvin | 14 | Bending | ✗ Not recommended |
| Diamond | 4 | Stretching | △ Limited |

---

## CONNECTION MECHANISMS

### Direct Match (Z=12 to Z=12):
```
Cell A (Z=12) ←→ Cell B (Z=12)
     ↓                  ↓
All 12 nodes connect directly to matching nodes
```
**Example**: Rhombic Dodecahedron ↔ Octet

### Hierarchical Distribution (Z=8 to Z=12):
```
Cell A (Z=8)  ←→ Cell B (Z=12)
     ↓                  ↓
Continuous distribution ensures connectivity
at shared boundary nodes via hierarchical arrangement
```
**Example**: BCC ↔ Octet

---

## DESIGN DECISION FLOWCHART

```
START: Select Application Requirements
    ↓
  High Stiffness Needed?
    ├─ YES → Use Octet as stretching cell
    │         ├─ Need energy absorption? → BCC + Octet ✓✓✓
    │         └─ Need perfect match? → Rhombic Dodec + Octet ✓✓✓
    │
    └─ NO → Consider FCC as stretching cell
              └─ Good toughness → BCC + FCC ✓✓
    ↓
  Verify Compatibility in Sidebar
    ↓
  Check Green "COMPATIBLE" Status
    ↓
  Proceed with Design
```

---

## TROUBLESHOOTING

### "No compatible stretching cells available"
**Cause**: You selected Kelvin cell (Z=14)
**Solution**: Switch to BCC or Rhombic Dodecahedron

### "WARNING: INCOMPATIBLE CELL COMBINATION"
**Cause**: Selected pairing not validated by research
**Solution**: Refer to compatible combinations above

### "Disconnected struts in visualization"
**Cause**: Incompatible Z values
**Solution**: Use only validated pairings from this guide

---

## RESEARCH PAPERS TO READ

**For BCC + Octet:**
1. Bian et al. (2023) - ACS Applied Materials & Interfaces
   DOI: 10.1021/acsami.2c20063

**For Rhombic Dodec + Octet:**
2. Yu et al. (2023) - Vibroengineering Procedia
   DOI: 10.21595/vp.2023.23454

**For Variable Density:**
3. Zhang et al. (2023) - Composite Structures
   DOI: 10.1016/j.compstruct.2022.116405

---

## SUMMARY

**✓ USE THESE:**
- BCC + Octet (most research support)
- BCC + FCC (good properties)
- Rhombic Dodecahedron + Octet (perfect Z match)

**✗ AVOID THESE:**
- Kelvin + Any
- Diamond + Any (unless specialized)
- Any unvalidated combinations

**⚠ ALWAYS CHECK:**
- Green "COMPATIBLE" status in sidebar
- Research references provided
- Nodal connectivity (Z values)

---

*This guide is based on peer-reviewed research as of 2023*
