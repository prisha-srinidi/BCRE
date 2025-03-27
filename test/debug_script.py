# Enhanced debug script
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

from src.enforcer import monolithic_enforcer, enforcer, computeEmptinessDict
from test.test_automata import CS3, CS4

# Create test automata
cs3 = CS3()
cs4 = CS4()
cs_intersection = monolithic_enforcer("CS3_CS4", CS3(), CS4())

# Get the emptiness dictionary
dict_enf = {}
if hasattr(cs_intersection, 'dictEnf'):
    dict_enf = cs_intersection.dictEnf
else:
    dict_enf = computeEmptinessDict(cs_intersection)

# Test problematic string
test = "acaabcba"

# Track transitions
prev_state = cs_intersection.q0
print(f"Initial state: {prev_state.name}")
print(f"  Component states: {prev_state.stateA.name}, {prev_state.stateB.name}")
print(f"  is_trap attribute: {hasattr(prev_state, 'is_trap') and prev_state.is_trap}")
print(f"  In dictEnf: {prev_state in dict_enf and dict_enf[prev_state]}")

for i, char in enumerate(test):
    next_state = cs_intersection.d(prev_state, char)
    
    # Test manual trap detection
    is_cs3_trap = next_state.stateA.name == 'T'
    is_cs4_trap = next_state.stateB.name == 'T'
    is_product_trap = hasattr(next_state, 'is_trap') and next_state.is_trap
    in_dict_trap = next_state in dict_enf and dict_enf[next_state]
    
    print(f"\nStep {i+1}: '{char}' -> {next_state.name}")
    print(f"  Component states: {next_state.stateA.name}, {next_state.stateB.name}")
    print(f"  CS3 trap: {is_cs3_trap}, CS4 trap: {is_cs4_trap}")
    print(f"  Product is_trap: {is_product_trap}, dictEnf: {in_dict_trap}")
    
    # Check what manual parallel composition would do
    cs3_next = cs3.d(prev_state.stateA, char)
    cs4_next = cs4.d(prev_state.stateB, char)
    manual_is_trap = cs3_next.name == 'T' or cs4_next.name == 'T'
    print(f"  Manual trap check: {manual_is_trap} (CS3: {cs3_next.name}, CS4: {cs4_next.name})")
    
    # Show what the enforcer function would do
    print(f"  Enforcer would suppress: {is_cs3_trap or is_cs4_trap or is_product_trap or in_dict_trap}")
    
    prev_state = next_state