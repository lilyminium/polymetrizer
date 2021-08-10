polymetrizer
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/lilyminium/polymetrizer/workflows/CI/badge.svg)](https://github.com/lilyminium/polymetrizer/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lilyminium/polymetrizer/branch/master/graph/badge.svg)](https://codecov.io/gh/lilyminium/polymetrizer/branch/master)


Generate force fields for polymer-like molecules

```python
from polymetrizer import Polymetrizer
from polymetrizer.tests.smiles import ALA, SER, CYS, ACE, NME
from openff.toolkit.typing.engines.smirnoff import ForceField

# ALA = "[H][N]([C@]([H])([C](=[O])[*:2])[C]([H])([H])[H])[*:1]"
# SER = "[H][O][C]([H])([H])[C@@]([H])([C](=[O])[*:2])[N]([H])[*:1]"
# CYS = "[H][S][C]([H])([H])[C@@]([H])([C](=[O])[*:2])[N]([H])[*:1]"
# ACE = "CC(=O)-[*:6]"
# NME = "[*:7]NC"

# The linkage points go HN-R1, O=C-R2
# so we need to bond R1-R2 to get a peptide bond
# ACE is O=C-R6
# NME is CN-R7

met = Polymetrizer(monomers=dict(Ser=SER, Ala=ALA, Cys=CYS, Ace=ACE, Nme=NME),
                   caps=[ACE, NME],
                   r_linkages = {1: {2, 6}, 7: {2}})
original_ff = ForceField("openff_unconstrained-1.3.0.offxml")
new_ff = met.polymetrize(original_ff,
                         n_neighbor_monomers=1,  # builds tripeptides
                         n_overlapping_atoms=3,
                         prune_isomorphs=False,
                         # minimize before doing AM1BCC charges with MMFF
                         minimize_geometry=True,
                         # optimization too expensive, but if so....
                         optimize_geometry=False,
                         optimize_method="m06-2x/def2-TZVP",
                        )     
offsmi = met.oligomers[-1].to_smiles()
offmol = Molecule.from_smiles(offsmi, allow_undefined_stereo=True)
# or offmol = met.oligomers[-1].to_openff()
new_ff.create_openmm_system(offmol.to_topology())
```

You can inspect the force field by ID:

```python
>>> new_ff.get_parameter_handler("LibraryCharges").parameters[:3]
[<LibraryChargeType with smirks: [#6:1](-[#6:2](=[#8:3])-[*])(-[#1:4])(-[#1:5])-[#1:6]  charge1: -0.17089466856904664 e  charge2: 0.6603946645360406 e  charge3: -0.6118399938641771 e  charge4: 0.0690333325963434 e  charge5: 0.0690333325963434 e  charge6: 0.0690333325963434 e  id: Ace_LibraryCharges_1  >,
 <LibraryChargeType with smirks: [*]-[#7:1](-[#6:2](-[#1:4])(-[#1:5])-[#1:6])-[#1:3]  charge1: -0.5661193370249459 e  charge2: 0.08167466576629978 e  charge3: 0.3122293353650381 e  charge4: 0.04634266700083117 e  charge5: 0.04634266700083117 e  charge6: 0.04634266700083117 e  id: Nme_LibraryCharges_2  >,
 <LibraryChargeType with smirks: [#1:1]-[#7:2](-[#6:3](-[#1:4])(-[#6:5](=[#8:6])-[*])-[#6:7](-[#1:8])(-[#1:9])-[#1:10])-[*]  charge1: 0.31730275763003457 e  charge2: -0.5472235705408015 e  charge3: 0.02803142861579617 e  charge4: 0.10330809572384587 e  charge5: 0.5753385718142998 e  charge6: -0.6132113829427004 e  charge7: -0.10350190449566951 e  charge8: 0.054191250524680594 e  charge9: 0.054191250524680594 e  charge10: 0.054191250524680594 e  id: Ala_LibraryCharges_3  >]
 
 >>> [x for x in new_ff.get_parameter_handler("ProperTorsions").parameters
 ... if "CysNme" in x.id]
 [<ProperTorsionType with smirks: [#1]-[#16]-[#6:1](-[#1])(-[#1])-[#6:2](-[#1])(-[#6:3](=[#8])-[#7:4](-[#6](-[#1])(-[#1])-[#1])-[#1])-[#7](-[#1])-[*]  periodicity1: 4  periodicity2: 2  phase1: 0.0 rad  phase2: 0.0 rad  id: CysNme_ProperTorsions_198  k1: 0.5270538172445449 kJ/mol  k2: 2.3552086237949736 kJ/mol  idivf1: 1.0  idivf2: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6:1](-[#1])(-[#6:2](=[#8])-[#7:3](-[#6:4](-[#1])(-[#1])-[#1])-[#1])-[#7](-[#1])-[*]  periodicity1: 2  periodicity2: 1  phase1: 3.141592653589793 rad  phase2: 0.0 rad  id: CysNme_ProperTorsions_199  k1: 7.540000637294281 kJ/mol  k2: 0.0 kJ/mol  idivf1: 1.0  idivf2: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6:1](-[#1])(-[#6:2](=[#8])-[#7:3](-[#6](-[#1])(-[#1])-[#1])-[#1:4])-[#7](-[#1])-[*]  periodicity1: 2  periodicity2: 1  phase1: 3.141592653589793 rad  phase2: 0.0 rad  id: CysNme_ProperTorsions_200  k1: 7.540000637294281 kJ/mol  k2: 0.0 kJ/mol  idivf1: 1.0  idivf2: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6:2](-[#1:1])(-[#6:3](=[#8])-[#7:4](-[#6](-[#1])(-[#1])-[#1])-[#1])-[#7](-[#1])-[*]  periodicity1: 3  phase1: 0.0 rad  id: CysNme_ProperTorsions_201  k1: -0.93912285044788 kJ/mol  idivf1: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6:1](=[#8])-[#7:2](-[#6:3](-[#1:4])(-[#1])-[#1])-[#1])-[#7](-[#1])-[*]  periodicity1: 2  periodicity2: 3  phase1: 0.0 rad  phase2: 0.0 rad  id: CysNme_ProperTorsions_202  k1: 0.7802448648863632 kJ/mol  k2: -0.19137745034585107 kJ/mol  idivf1: 1.0  idivf2: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6:1](=[#8])-[#7:2](-[#6:3](-[#1])(-[#1:4])-[#1])-[#1])-[#7](-[#1])-[*]  periodicity1: 2  periodicity2: 3  phase1: 0.0 rad  phase2: 0.0 rad  id: CysNme_ProperTorsions_203  k1: 0.7802448648863632 kJ/mol  k2: -0.19137745034585107 kJ/mol  idivf1: 1.0  idivf2: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6:1](=[#8])-[#7:2](-[#6:3](-[#1])(-[#1])-[#1:4])-[#1])-[#7](-[#1])-[*]  periodicity1: 2  periodicity2: 3  phase1: 0.0 rad  phase2: 0.0 rad  id: CysNme_ProperTorsions_204  k1: 0.7802448648863632 kJ/mol  k2: -0.19137745034585107 kJ/mol  idivf1: 1.0  idivf2: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6:2](=[#8:1])-[#7:3](-[#6:4](-[#1])(-[#1])-[#1])-[#1])-[#7](-[#1])-[*]  periodicity1: 2  periodicity2: 1  phase1: 3.141592653589793 rad  phase2: 0.0 rad  id: CysNme_ProperTorsions_205  k1: 13.616417454727802 kJ/mol  k2: 0.0 kJ/mol  idivf1: 1.0  idivf2: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6](-[#1])(-[#6:2](=[#8:1])-[#7:3](-[#6](-[#1])(-[#1])-[#1])-[#1:4])-[#7](-[#1])-[*]  periodicity1: 2  periodicity2: 1  phase1: 3.141592653589793 rad  phase2: 0.0 rad  id: CysNme_ProperTorsions_206  k1: 7.947573398773489 kJ/mol  k2: 3.062935162607704 kJ/mol  idivf1: 1.0  idivf2: 1.0  >,
 <ProperTorsionType with smirks: [#1]-[#16]-[#6](-[#1])(-[#1])-[#6:2](-[#1])(-[#6:3](=[#8])-[#7:4](-[#6](-[#1])(-[#1])-[#1])-[#1])-[#7:1](-[#1])-[*]  periodicity1: 1  periodicity2: 2  phase1: 3.141592653589793 rad  phase2: 3.141592653589793 rad  id: CysNme_ProperTorsions_207  k1: -0.25777191731403987 kJ/mol  k2: 2.923283511411457 kJ/mol  idivf1: 1.0  idivf2: 1.0  >]
```

### Copyright

Copyright (c) 2021, Lily Wang


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
