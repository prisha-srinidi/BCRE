#!/usr/bin/env python3
"""
test_automata.py

This file defines example automata for testing runtime enforcement with three approaches:
1. Individual property enforcement
2. Monolithic enforcer (product construction)
3. Parallel enforcement via longest common subsequence

The automata tested are:
- CS3: The first 3 actions must be: 'a', any letter from {a,b,c}, 'b'
- CS4: The first 4 actions must be: 'a', any letter from {a,b,c}, 'b', 'c'


"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

from src.enforcer import state, DFA, monolithic_enforcer, enforcer, longest_common_subsequence, serial_enforcer

# Define the alphabet for these properties
alphabet = list("abc")

#############################
# Automaton for CS3 property
#############################
def CS3(Type="DFA"):
    # States:
    # S0: initial, waiting for first action.
    # S1: after receiving 'a' as first action.
    # S2: after receiving second action (any letter in alphabet).
    # S3: after receiving third action 'b' -> accepting.
    # T: trap state.
    S0 = state('S0')
    S1 = state('S1')
    S2 = state('S2')
    S3 = state('S3')  # accepting state
    T  = state('T')   # trap state

    # From S0: only 'a' is allowed; any other input goes to T.
    for a in alphabet:
        if a == 'a':
            S0.transit[a] = S1
        else:
            S0.transit[a] = T

    # From S1: any letter in {a, b, c} goes to S2.
    for a in alphabet:
        S1.transit[a] = S2

    # From S2: only 'b' is allowed to go to S3.
    for a in alphabet:
        if a=='b':
            S2.transit[a] = S3
        else:
            S2.transit[a] = T

    # From S3 (accepting): remain in S3 for any input.
    for a in alphabet:
        S3.transit[a] = S3

    # Trap state T loops to itself on all inputs.
    for a in alphabet:
        T.transit[a] = T

    # Transition and acceptance functions
    d = lambda state, a: state.transit[a]
    F = lambda state: state in [S3]
    
    return DFA('CS3', alphabet, [S0, S1, S2, S3, T], S0, F, d, [S3])

#############################
# Automaton for CS4 property
#############################
def CS4(Type="DFA"):
    # States:
    # S0: initial.
    # S1: after receiving 'a'.
    # S2: after receiving second action (any letter from alphabet).
    # S3: after receiving 'b'.
    # S4: after receiving 'c' -> accepting.
    # T: trap state.
    S0 = state('S0')
    S1 = state('S1')
    S2 = state('S2')
    S3 = state('S3')
    S4 = state('S4')  # accepting state
    T  = state('T')   # trap state

    # From S0: only 'a' is allowed.
    for a in alphabet:
        if a == 'a':
            S0.transit[a] = S1
        else:
            S0.transit[a] = T

    # From S1: any letter in {a, b, c} goes to S2.
    for a in alphabet:
        S1.transit[a] = S2

    # From S2: only 'b' is allowed.
    for a in alphabet:
        if a == 'b':
            S2.transit[a] = S3
        else:
            S2.transit[a] = T

    # From S3: only 'c' is allowed.
    for a in alphabet:
        if a == 'c':
            S3.transit[a] = S4
        else:
            S3.transit[a] = T

    # S4 is accepting; once reached, any input loops in S4.
    for a in alphabet:
        S4.transit[a] = S4

    # Trap state T loops to itself.
    for a in alphabet:
        T.transit[a] = T

    # Transition and acceptance functions
    d = lambda state, a: state.transit[a]
    F = lambda state: state in [S4]
    
    return DFA('CS4', alphabet, [S0, S1, S2, S3, S4, T], S0, F, d, [S4])

#############################
# Automaton for CS4 property
#############################
def CS4(Type="DFA"):
    # States:
    # S0: initial.
    # S1: after receiving 'a'.
    # S2: after receiving second action (any letter from alphabet).
    # S3: after receiving 'b'.
    # S4: after receiving 'c' -> accepting.
    # T: trap state.
    S0 = state('S0')
    S1 = state('S1')
    S2 = state('S2')
    S3 = state('S3')
    S4 = state('S4')  # accepting state
    T  = state('T')   # trap state

    # From S0: only 'a' is allowed.
    for a in alphabet:
        if a == 'a':
            S0.transit[a] = S1
        else:
            S0.transit[a] = T

    # From S1: any letter in {a, b, c} goes to S2.
    for a in alphabet:
        S1.transit[a] = S2

    # From S2: only 'b' is allowed.
    for a in alphabet:
        if a == 'b':
            S2.transit[a] = S3
        else:
            S2.transit[a] = T

    # From S3: only 'c' is allowed.
    for a in alphabet:
        if a == 'c':
            S3.transit[a] = S4
        else:
            S3.transit[a] = T

    # S4 is accepting; once reached, any input loops in S4.
    for a in alphabet:
        S4.transit[a] = S4

    # Trap state T loops to itself.
    for a in alphabet:
        T.transit[a] = T

    # Transition and acceptance functions
    d = lambda state, a: state.transit[a]
    F = lambda state: state in [S4]
    
    return DFA('CS4', alphabet, [S0, S1, S2, S3, S4, T], S0, F, d, [S4])



#############################
# Main: Testing the Automata
#############################
if __name__ == '__main__':
    print("Testing enforcement approaches on CS3 and CS4 properties")
    
    # Create the automata
    cs3 = CS3()
    cs4 = CS4()
    
    # Test strings for CS3 and CS4
    cs_test_strings = [
        "a",       
        "ac",      
        "aca",     
        "acaa",    
        "acaab",      
        "acaabc",      
        "acaabcb",  
        "acaabcba" 
    ]
    
    # Buffer size to use for all tests
    buffer_size = 3
    
    # Create the monolithic enforcer for product construction
    cs_intersection = monolithic_enforcer("CS3_CS4", CS3(), CS4())
    
    # Test each input string with the three enforcement approaches
    print("\nComparing enforcement approaches:")
    print("================================")
    
    for test in cs_test_strings:
        print(f"\nInput: '{test}'")
        
        # 1. Individual property enforcement
        cs3_output = enforcer(cs3, list(test), buffer_size)
        cs4_output = enforcer(cs4, list(test), buffer_size)
        
        # 2. Monolithic enforcement
        monolithic_output = enforcer(cs_intersection, list(test), buffer_size)
        
        # 3. Parallel enforcement with LCS
        output_cs3 = ''.join(cs3_output)
        output_cs4 = ''.join(cs4_output)
        lcs_output = longest_common_subsequence(output_cs3, output_cs4)
        
        # 4. Serial enforcement (both directions)
        cs3_then_cs4 = serial_enforcer("CS3_THEN_CS4", cs3, cs4)
        cs4_then_cs3 = serial_enforcer("CS4_THEN_CS3", cs4, cs3)
        serial_output_3_4, _ = cs3_then_cs4(test, buffer_size)
        serial_output_4_3, _ = cs4_then_cs3(test, buffer_size)
        
        # Print results for comparison
        print(f"  CS3 Individual: '{output_cs3}'")
        print(f"  CS4 Individual: '{output_cs4}'")
        print(f"  Monolithic: '{''.join(monolithic_output)}'")
        print(f"  Parallel LCS: '{lcs_output}'")
        print(f"  Serial (CS3→CS4): '{''.join(serial_output_3_4)}'")
        print(f"  Serial (CS4→CS3): '{''.join(serial_output_4_3)}'")
        
        # Check if monolithic and serial outputs match
        if ''.join(monolithic_output) == ''.join(serial_output_3_4):
            print("  ✓ Monolithic and Serial(CS3→CS4) outputs match")
        else:
            print("  ✗ Monolithic and Serial(CS3→CS4) outputs differ")
            
        if ''.join(monolithic_output) == ''.join(serial_output_4_3):
            print("  ✓ Monolithic and Serial(CS4→CS3) outputs match")
        else:
            print("  ✗ Monolithic and Serial(CS4→CS3) outputs differ")
            
        # Check if monolithic and parallel LCS outputs match
        if ''.join(monolithic_output) == lcs_output:
            print("  ✓ Monolithic and Parallel LCS outputs match")
        else:
            print("  ✗ Monolithic and Parallel LCS outputs differ")
            
    # # Continue with the rest of your existing code
    # complex_input = "acaabcabcabacbaacbaabc"
    
    # # Add the serial enforcement test for the complex input
    # print("\nSerial Enforcement with Complex Input:")
    # cs3_then_cs4 = serial_enforcer("CS3_THEN_CS4", cs3, cs4)
    # cs4_then_cs3 = serial_enforcer("CS4_THEN_CS3", cs4, cs3)
    # serial_output_3_4, _ = cs3_then_cs4(complex_input, buffer_size)
    # serial_output_4_3, _ = cs4_then_cs3(complex_input, buffer_size)
    # print(f"  Serial (CS3→CS4): '{''.join(serial_output_3_4)}'")
    # print(f"  Serial (CS4→CS3): '{''.join(serial_output_4_3)}'")
    
    # # Extended analysis
    # print("\nExtended Output Analysis:")
    # print("  CS3 output length: ", len(output_cs3))
    # print("  CS4 output length: ", len(output_cs4))
    # print("  Monolithic output length: ", len(''.join(monolithic_output)))
    # print("  Parallel LCS output length: ", len(lcs_output))
    # print("  Serial (CS3→CS4) output length: ", len(''.join(serial_output_3_4)))
    # print("  Serial (CS4→CS3) output length: ", len(''.join(serial_output_4_3)))
