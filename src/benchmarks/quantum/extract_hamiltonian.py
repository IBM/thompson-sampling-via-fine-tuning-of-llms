import numbers
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner
from openfermion.utils import count_qubits
from openfermion.ops import QubitOperator

def format_hamiltonian(hamiltonian, max_terms=20):
    #Print first few Pauli terms
    print(f"{len(hamiltonian.terms.items())} pauli components of Hamiltonian")
    pauli_terms = []
    num_qubits = count_qubits(hamiltonian)
    for i, (term, coeff) in enumerate(sorted(hamiltonian.terms.items(), key=lambda item: abs(item[1]), reverse=True)):
        if i >= max_terms:
            break
        pauli_string = ["I"] * num_qubits
        for pauli in term:
            pauli_string[pauli[0]] = pauli[1]
        coeff = coeff if isinstance(coeff, numbers.Number) else coeff.item()
        pauli_terms.append(("".join(pauli_string), coeff))
    print(pauli_terms)
    print("pauli_terms = [")
    for paulis, coeff in pauli_terms:
        print(f"    (\"{paulis}\", {coeff}),")
    print("]")
    print(f"the sum of the absolute value of the {len(pauli_terms)} Pauli terms is {sum(abs(coef) for _, coef in pauli_terms)}")

def chemistry(geometry):
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    # Run electronic structure calculation
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    # Convert to qubit Hamiltonian
    fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
    qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    format_hamiltonian(qubit_hamiltonian, max_terms=50)

def transverse_field_ising(n_qubits, J, h, max_terms):
    # Build TFIM Hamiltonian: H = -J Σ Z_i Z_{i+1} - h Σ X_i
    hamiltonian = QubitOperator()
    # Z coupling (dipolar interaction in magnets favoring alignment)
    for i in range(n_qubits - 1):
        term = QubitOperator(f'Z{i} Z{i+1}', -J)
        hamiltonian += term
    # X transverse field (transverse magnetic field in magnets favoring superposition)
    for i in range(n_qubits):
        term = QubitOperator(f'X{i}', -h)
        hamiltonian += term
    format_hamiltonian(hamiltonian)

#1	Start from molecular electronic Hamiltonian (fermionic operators) in STO-3G basis for H2 at 1.5 angstroms distance
#2	Map fermionic operators to qubit operators using Jordan-Wigner mapping.
#3	Result: a 4-qubit Hamiltonian with Pauli operators and coefficients. Each qubit corresponds to occupancy of a spin orbital.
h2_pauli_terms = [
    ("IIII", -0.4917857773035379),
    ("IIZZ", 0.14585519030093108),
    ("ZIIZ", 0.13992103890325305),
    ("IZZI", 0.13992103890325305),
    ("ZZII", 0.13817584576560327),
    ("ZIII", 0.09345649667701597),
    ("IZII", 0.09345649667701597),
    ("ZIZI", 0.08253705488832758),
    ("IZIZ", 0.08253705488832758),
    ("XXYY", -0.057383984014925477),
    ("XYYX", 0.057383984014925477),
    ("YXXY", 0.057383984014925477),
    ("YYXX", -0.057383984014925477),
    ("IIZI", -0.03564481621009499),
    ("IIIZ", -0.03564481621009499),
]

#1	Start from molecular electronic Hamiltonian (fermionic operators) in STO-3G basis for H2O at equilibrium geometry (based on Hartree-Fock STO-3G simulation at https://cccbdb.nist.gov/geom3x.asp?method=1&basis=20)
#2	Map fermionic operators to qubit operators using Jordan-Wigner mapping.
#3	Result: a 14-qubit Hamiltonian with Pauli operators and coefficients. Each qubit corresponds to occupancy of a spin orbital.
h2o_pauli_terms = [
    ("IIIIIIIIIIIIII", -46.53906811218841),
    ("ZIIIIIIIIIIIII", 12.411383637970662),
    ("IZIIIIIIIIIIII", 12.411383637970662),
    ("IIZIIIIIIIIIII", 1.6495661300966775),
    ("IIIZIIIIIIIIII", 1.6495661300966773),
    ("IIIIIIIIZIIIII", 1.366626079112477),
    ("IIIIIIIIIZIIII", 1.366626079112477),
    ("IIIIIIZIIIIIII", 1.2973670561264576),
    ("IIIIIIIZIIIIII", 1.2973670561264576),
    ("IIIIZIIIIIIIII", 1.1907977181056395),
    ("IIIIIZIIIIIIII", 1.1907977181056393),
    ("ZZIIIIIIIIIIII", 1.186278662509342),
    ("IIIIIIIIIIIIZI", 0.8108693281499294),
    ("IIIIIIIIIIIIIZ", 0.8108693281499294),
    ("IIIIIIIIIIZIII", 0.7929139834447949),
    ("IIIIIIIIIIIZII", 0.7929139834447949),
    ("IIIIXZZZZZZZXI", -0.36912842190139533),
    ("IIIIYZZZZZZZYI", -0.36912842190139533),
    ("IIIIIXZZZZZZZX", -0.36912842190139533),
    ("IIIIIYZZZZZZZY", -0.36912842190139533),
    ("ZIIIIIIIIZIIII", 0.27883553960810226),
    ("IZIIIIIIZIIIII", 0.27883553960810226),
    ("IIXZZZZZZZXIII", -0.276451706575761),
    ("IIYZZZZZZZYIII", -0.276451706575761),
    ("IIIXZZZZZZZXII", -0.276451706575761),
    ("IIIYZZZZZZZYII", -0.276451706575761),
    ("ZIIIIIIIZIIIII", 0.2723298090223371),
    ("IZIIIIIIIZIIII", 0.2723298090223371),
    ("ZIIZIIIIIIIIII", 0.25159119439498745),
    ("IZZIIIIIIIIIII", 0.25159119439498745),
    ("ZIIIIIIZIIIIII", 0.2450615307513267),
    ("IZIIIIZIIIIIII", 0.2450615307513267),
    ("IIIIIIIXZZZXII", 0.2395124897722309),
    ("IIIIIIIYZZZYII", 0.2395124897722309),
    ("IIIIIIXZZZXIII", 0.23951248977223089),
    ("IIIIIIYZZZYIII", 0.23951248977223089),
    ("ZIIIIIZIIIIIII", 0.23827947709212965),
    ("IZIIIIIZIIIIII", 0.23827947709212965),
    ("ZIZIIIIIIIIIII", 0.23693930172228894),
    ("IZIZIIIIIIIIII", 0.23693930172228894),
    ("IIIIIIIIZZIIII", 0.22003977334376112),
    ("ZIIIIIIIIIIIIZ", 0.21368080059155664),
    ("IZIIIIIIIIIIZI", 0.21368080059155664),
    ("ZIIIIIIIIIIIZI", 0.20857094958056188),
    ("IZIIIIIIIIIIIZ", 0.20857094958056188),
    ("ZIIIIIIIIIIZII", 0.20186132478332114),
    ("IZIIIIIIIIZIII", 0.20186132478332114),
    ("ZIIIIZIIIIIIII", 0.19872476385742224),
    ("IZIIZIIIIIIIII", 0.19872476385742224),
    ("ZIIIZIIIIIIIII", 0.19595785190719078),
]

# Transverse-Field Ising Model of ferromagnetism (and other phenomena)
ising_pauli_terms = [
    ("ZZIIIIII", -1.0),
    ("IZZIIIII", -1.0),
    ("IIZZIIII", -1.0),
    ("IIIZZIII", -1.0),
    ("IIIIZZII", -1.0),
    ("IIIIIZZI", -1.0),
    ("IIIIIIZZ", -1.0),
    ("XIIIIIII", -1.0),
    ("IXIIIIII", -1.0),
    ("IIXIIIII", -1.0),
    ("IIIXIIII", -1.0),
    ("IIIIXIII", -1.0),
    ("IIIIIXII", -1.0),
    ("IIIIIIXI", -1.0),
    ("IIIIIIIX", -1.0),
]


# Consider the field of quantum sensing, metrology, or calibration of a quantum state for an experiment. Assume for instance
# - qubit 1 is an ancilla qubit that needs to be perfectly initialized to be |0>
# - qubit 2 is a control qubit that needs to be set to |1>
# - qubit 3, and 4 are part of an interferometric setup and each need to be in a superposition with correct phase to enable a specific type of interaction
# - qubit 5 and 6 should be entangled in a specific way to the Bell state (|00> + |11>)/√2, for instance for application in an error-correcting circuit
# Moreover, suppose you interact with an unknown black-box system (possibly a physical system connected to the quantum computer). Then one has to resort to Bayesian optimization to find the quantum circuit preparing a low-energy state.

artificial_pauli_terms = [
   ("ZIIIIII", -1.0), # ideally qubit |0>
   ("IZIIIII", 1.0),  # ideally qubit |1>
   ("IIXIIII", -1.0), # ideally qubit |+> = |0> + |1>
   ("IIIYIII", -1.0), # ideally qubit |i> = |0> + i |1>
   ("IIIIZZI", -1.0), # ideally qubits in a superposition of |00> and |11>
   ("IIIIXXI", -1.0), # ideally qubits in a superposition of |++> or |-->
   ("IZZIIII", -1.0), # ideally qubits |00> or |11>, without this term all paulis would be simultaneously diagonalizable and a single shared hamiltonian could be found
   ("YIIIIIX", -0.5), # non-ideality
   ("IIZIIIX", -0.5), # non-ideality
]


if __name__ == "__main__":
    # Define molecule
    LiH_geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.45))]
    H2O_geometry = [ # geometry based on Hartree-Fock STO-3G simulation at https://cccbdb.nist.gov/geom3x.asp?method=1&basis=20
        ('O', (0.0, 0.0, 0.1272)),
        ('H', (0.0, 0.7581, -0.5086)),
        ('H', (0.0, -0.7581, -0.5086))
    ]
    H2_geometry = [('H', (0, 0, 0)), ('H', (0, 0, 1.5))]

    chemistry(H2O_geometry)
    transverse_field_ising(n_qubits=8, J=1.0, h=1.0, max_terms=20)