# TEME-ECT-PINN-Laplace

**Supplementary materials for:**

> Agudelo Cárdenas, A., Peña, C., Forero, N., & Rodríguez Carmona, E. (2025).
> *From White-Box to Epistemic Agent: Teaching Laplace's Equation via
> Low-Cost Experiments and Physics-Informed Neural Networks.*
> [Journal — under review]

**DOI:** [to be assigned upon acceptance]
**ORCID (corresponding author):** [0000-0003-0598-2317](https://orcid.org/0000-0003-0598-2317)
**Contact:** alexander.cardenas@esing.edu.co

---

## Repository Structure

```
TEME-ECT-PINN-Laplace/
│
├── README.md                                   ← this file
├── LICENSE                                     ← MIT
│
├── S1_octave/
│   └── a1114b.m                                ← S1: GNU Octave λ-Fourier code
│
├── S2_pinn/
│   └── pinn_laplace_numpy.py                   ← S2: Python PINN implementation
│
├── S3_data/
│   └── ExcelConfiguracionesElectroest_tica.xlsx ← S3: Experimental data (5 configs)
│
└── S4_student_report/
    └── InformeMath3-Gomez_Guzman.pdf           ← S4: Student IEEE-format report
```

---

## Supplementary Files

### S1 — GNU Octave code (`a1114b.m`)
Complete implementation of the four-stage Néstor λ-corrected Fourier series
solution for the parallel-bar configuration.
- Parameters: a = 16 cm, b = 8 cm, λ = 0.029940 cm⁻², N = 30 terms
- Produces 3D surface visualization and comparison with experimental data
- Author: N. Forero (co-author), adapted from student implementation

### S2 — Python PINN code (`pinn_laplace_numpy.py`)
Reproducible NumPy-only implementation of the Physics-Informed Neural Network
for all three electrode configurations.

**Dependencies:**
```
NumPy  >= 1.24
SciPy  >= 1.10
Matplotlib >= 3.7
```

**Usage:**
```bash
python pinn_laplace_numpy.py --config parallel --plot
python pinn_laplace_numpy.py --config horseshoe --plot
python pinn_laplace_numpy.py --config LL --plot
python pinn_laplace_numpy.py --config all
```

**Expected results:**

| Configuration  | MAPE   | R²    | MAE (V) | Epochs |
|----------------|--------|-------|---------|--------|
| Parallel bars  | 6.69%  | 0.954 | 0.047   | 800    |
| Horseshoe      | 22.27% | 0.760 | 0.140   | 1500   |
| L-L perimetral | 30.2%  | 0.594 | 0.198   | 1000   |

### S3 — Experimental data (`ExcelConfiguracionesElectroest_tica.xlsx`)
Complete experimental measurement matrices for all five electrode configurations
(sheets: Signos, Paralelas, L, Herradura, Semicirculo) as collected by student
learning communities, ESING 2024.

**Data collectors:** Gómez Villamizar, A.G. & Guzmán Cruz, M.A. (ESING, 2024)

### S4 — Student report (`InformeMath3-Gomez_Guzman.pdf`)
Selected pages from the IEEE-format student learning community report (2024),
including experimental photographs, QuickField and FEMM visualizations, and
the student-written Análisis y Conclusiones section.

---

## Experimental Setup

- **Apparatus:** Ceramic refractory tank, 250 ml saline (300 g NaCl), εᵣ ≈ 78.5
- **Source:** 1.5 V DC regulated supply
- **Measurement:** Digital multimeter ±0.001 V (±0.07% at full scale)
- **Total cost:** ~USD 26 (replicable in any institution worldwide)
- **Grid:** 144 nodes (9×16) for parallel bars; 171 nodes (9×19) for L-L and horseshoe

---

## Five-Method Progression (TEME Cycle)

| Method | Type       | Error (parallel bars) | Key epistemic finding          |
|--------|------------|----------------------|-------------------------------|
| M1     | White-box  | 17.87%               | Voltage paradox activated      |
| M2     | Gray-box   | **6.83% ✓**          | Simplicity Paradox (M2 < M3)   |
| M3     | White-box  | 21.86%               | Confirms boundary hypothesis   |
| M4     | Gray-box   | 32.38%               | Inverse failure diagnostic     |
| M5     | Hybrid     | **6.69% ★**          | Epistemic agent (PINN)         |

**Simplicity Paradox:** M2 (6.83%) outperforms M3 (21.86%) by a factor of 3.2 —
precise boundary knowledge surpasses computational sophistication.

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{agudelocardenas2025teme,
  title   = {From White-Box to Epistemic Agent: Teaching Laplace's Equation
             via Low-Cost Experiments and Physics-Informed Neural Networks},
  author  = {Agudelo C{\'a}rdenas, Alexander and Pe{\~n}a, Carlos and
             Forero, N{\'e}stor and Rodr{\'i}guez Carmona, Esperanza},
  journal = {[Journal name — under review]},
  year    = {2025},
  doi     = {[to be assigned]}
}
```

---

## License

Code: MIT License (see LICENSE file)
Data and figures: CC BY 4.0

---

*This work was conducted at ESING (Escuela de Ingenieros Militares) and
Universidad Militar Nueva Granada, Bogotá, Colombia, 2024.*
