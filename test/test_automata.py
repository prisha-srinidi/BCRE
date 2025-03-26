#!/usr/bin/env python3
"""
test_automata.py

This file defines example automata for testing runtime enforcement.

Original automata:
    CS3: The first 3 actions must be: 'a', any letter from {a,b,c}, 'b' (then remain accepting)
    CS4: The first 4 actions must be: 'a', any letter from {a,b,c}, 'b', 'c' (then remain accepting)

Added automata:
    RE1: a (b|c)* a b* - Accepts strings starting with 'a', followed by any number of 'b' or 'c',
         followed by 'a', and ending with any number of 'b's.
    RE2: a (b|c)* a - Accepts strings starting with 'a', followed by any number of 'b' or 'c',
         and ending with 'a'.

The intersection RE1 ∩ RE2 should be: a (b|c)* a
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

from src.enforcer import state, DFA, monolithic_enforcer

# Define the alphabet for these properties
alphabet = list("abc")

# Existing CS3 and CS4 automata definitions remain unchanged
#############################
# Automaton for CS3 property
#############################
def CS3(Type="DFA"):
    # States:
    # S0: initial, waiting for first action.
    # S1: after receiving 'a' as first action.
    # S2: after receiving second action (any letter in alphabet).
    # S3: after receiving third action (any letter in alphabet) -> accepting.
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

    # From S2: any letter in {a, b, c} goes to S3.
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

    # For simplicity, let the transition function d simply look up the next state in the transit dictionary.
    d = lambda state, a: state.transit[a]
    # F is a predicate that returns True if the state is accepting.
    F = lambda state: state in [S3]
    # Return DFA with:
    #   S = alphabet, Q = list of states, q0 = S0, F = F, d = d, end = [S3]
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

    # Transition function d: uses the state's transit dictionary.
    d = lambda state, a: state.transit[a]
    # Acceptance predicate: returns True if the state is S4.
    F = lambda state: state in [S4]
    return DFA('CS4', alphabet, [S0, S1, S2, S3, S4, T], S0, F, d, [S4])


#############################
# Automaton for RE1: a (b|c)* a b*
#############################
def RE1(Type="DFA"):
    # States:
    # S0: initial state
    # S1: after receiving first 'a'
    # S2: after receiving second 'a' (accepting)
    # T: trap state
    S0 = state('S0')
    S1 = state('S1')
    S2 = state('S2')  # accepting state
    T  = state('T')   # trap state

    # From S0: only 'a' is allowed to move to S1; others go to trap
    for a in alphabet:
        if a == 'a':
            S0.transit[a] = S1
        else:
            S0.transit[a] = T

    # From S1: 'b' or 'c' loop back to S1, 'a' goes to S2
    for a in alphabet:
        if a == 'a':
            S1.transit[a] = S2
        elif a in ['b', 'c']:
            S1.transit[a] = S1
        else:
            S1.transit[a] = T

    # From S2 (accepting): 'b' loops back to S2, others go to trap
    for a in alphabet:
        if a == 'b':
            S2.transit[a] = S2
        else:
            S2.transit[a] = T

    # Trap state T loops to itself
    for a in alphabet:
        T.transit[a] = T

    # Transition function and acceptance predicate
    d = lambda state, a: state.transit[a]
    F = lambda state: state in [S2]
    
    return DFA('RE1', alphabet, [S0, S1, S2, T], S0, F, d, [S2])


#############################
# Automaton for RE2: a (b|c)* a
#############################
def RE2(Type="DFA"):
    # States:
    # S0: initial state
    # S1: after receiving first 'a'
    # S2: after receiving second 'a' (accepting)
    # T: trap state
    S0 = state('S0')
    S1 = state('S1')
    S2 = state('S2')  # accepting state
    T  = state('T')   # trap state

    # From S0: only 'a' is allowed to move to S1; others go to trap
    for a in alphabet:
        if a == 'a':
            S0.transit[a] = S1
        else:
            S0.transit[a] = T

    # From S1: 'b' or 'c' loop back to S1, 'a' goes to accepting state S2
    for a in alphabet:
        if a == 'a':
            S1.transit[a] = S2
        elif a in ['b', 'c']:
            S1.transit[a] = S1
        else:
            S1.transit[a] = T

    # From S2 (accepting): all inputs go to trap (we want to end with 'a')
    for a in alphabet:
        S2.transit[a] = T

    # Trap state T loops to itself
    for a in alphabet:
        T.transit[a] = T

    # Transition function and acceptance predicate
    d = lambda state, a: state.transit[a]
    F = lambda state: state in [S2]
    
    return DFA('RE2', alphabet, [S0, S1, S2, T], S0, F, d, [S2])


#############################
# Main: Testing the Automata
#############################
if __name__ == '__main__':
    print("Testing individual properties:")
    
    cs3 = CS3()
    cs4 = CS4()
    
    # Revised test strings for CS3 and CS4 with only valid alphabet characters
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
    
    print("\nCS3 property (a,*,b,*):")
    print("-------------------------")
    for test in cs_test_strings:
        # Process with trap state suppression for CS3
        enforced = ''
        current = cs3.q0
        for char in test:
            next_state = cs3.d(current, char)
            # Check if next_state is a trap state
            if next_state in [state for state in cs3.Q if state.name == 'T']:
                # Suppress the event that leads to trap state
                continue
            else:
                # Add char to enforced output and update current state
                enforced += char
                current = next_state
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    print("\nCS4 property (a,*,b,c,*):")
    print("--------------------------")
    for test in cs_test_strings:
        # Process with trap state suppression for CS4
        enforced = ''
        current = cs4.q0
        for char in test:
            next_state = cs4.d(current, char)
            # Check if next_state is a trap state
            if next_state in [state for state in cs4.Q if state.name == 'T']:
                # Suppress the event that leads to trap state
                continue
            else:
                # Add char to enforced output and update current state
                enforced += char
                current = next_state
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    # Testing CS3 ∩ CS4 intersection with progressive inputs
    print("\nTesting Intersection (CS3 ∩ CS4):")
    print("----------------------------------")

    for test in cs_test_strings:
        # Manual computation of intersection output with proper trap state detection
        enforced = ''
        # Use the SAME automaton instances you already created
        current_cs3 = cs3.q0  # Use existing automaton instance
        current_cs4 = cs4.q0  # Use existing automaton instance
        
        for char in test:
            next_state_cs3 = cs3.d(current_cs3, char)
            next_state_cs4 = cs4.d(current_cs4, char)
            
            # Check if either leads to a trap state
            is_trap_cs3 = next_state_cs3 in [state for state in cs3.Q if state.name == 'T']
            is_trap_cs4 = next_state_cs4 in [state for state in cs4.Q if state.name == 'T']
            
            if not is_trap_cs3 and not is_trap_cs4:
                enforced += char
                current_cs3 = next_state_cs3
                current_cs4 = next_state_cs4
        
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    # Testing the new RE1 and RE2 automata
    re1 = RE1()
    re2 = RE2()
    
    # Progressive test strings for 'accab' showing each step
    re_progressive_strings = [
        "a",    # First character
        "ac",   # First two characters
        "acc",  # First three characters
        "acca", # First four characters
        "accab" # Full test string
    ]
    
    print("\nRE1 property (a (b|c)* a b*):")
    print("-----------------------------")
    for test in re_progressive_strings:
        # Process with trap state suppression for RE1
        enforced = ''
        current = re1.q0
        for char in test:
            next_state = re1.d(current, char)
            if next_state in [state for state in re1.Q if state.name == 'T']:
                # Suppress the event that leads to trap state
                continue
            else:
                enforced += char
                current = next_state
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    print("\nRE2 property (a (b|c)* a):")
    print("-------------------------")
    for test in re_progressive_strings:
        # Process with trap state suppression for RE2
        enforced = ''
        current = re2.q0
        for char in test:
            next_state = re2.d(current, char)
            if next_state in [state for state in re2.Q if state.name == 'T']:
                # Suppress the event that leads to trap state
                continue
            else:
                enforced += char
                current = next_state
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    # Test RE1 ∩ RE2 intersection
    print("\nTesting Intersection (RE1 ∩ RE2) - Progressive Testing of 'accab':")
    print("------------------------------------------------------------------")

    for test in re_progressive_strings:
        enforced = ''
        current_re1 = re1.q0  # Use existing automaton instance
        current_re2 = re2.q0  # Use existing automaton instance
        
        for char in test:
            next_state_re1 = re1.d(current_re1, char) 
            next_state_re2 = re2.d(current_re2, char)
            
            # Check if either leads to a trap state
            is_trap_re1 = next_state_re1 in [state for state in re1.Q if state.name == 'T']
            is_trap_re2 = next_state_re2 in [state for state in re2.Q if state.name == 'T']
            
            if not is_trap_re1 and not is_trap_re2:
                enforced += char
                current_re1 = next_state_re1
                current_re2 = next_state_re2
        
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    # More test cases for RE1 and RE2
    re_test_strings = [
        "aa",      # Accepted by both RE1 and RE2
        "aba",     # Accepted by both RE1 and RE2
        "acba",    # Accepted by both RE1 and RE2
        "aab",     # Accepted by RE1, not by RE2
        "acab",    # Accepted by RE1, not by RE2
        "abcaa",   # Not accepted by RE1, not by RE2
        "abca",    # Accepted by both RE1 and RE2
        "abcabb"   # Accepted by RE1, not by RE2
    ]
    
    print("\nAdditional Test Cases for RE1 (a (b|c)* a b*):")
    print("--------------------------------------------")
    for test in re_test_strings:
        # Process output for RE1
        enforced = ''
        current = re1.q0
        for char in test:
            next_state = re1.d(current, char)
            if next_state in [state for state in re1.Q if state.name == 'T']:
                # Suppress the event that leads to trap state
                continue
            else:
                enforced += char
                current = next_state
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    print("\nAdditional Test Cases for RE2 (a (b|c)* a):")
    print("------------------------------------------")
    for test in re_test_strings:
        # Process output for RE2
        enforced = ''
        current = re2.q0
        for char in test:
            next_state = re2.d(current, char)
            if next_state in [state for state in re2.Q if state.name == 'T']:
                # Suppress the event that leads to trap state
                continue
            else:
                enforced += char
                current = next_state
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    print("\nAdditional Test Cases for Intersection (RE1 ∩ RE2):")
    print("-------------------------------------------------")
    for test in re_test_strings:
        enforced = ''
        current_re1 = re1.q0
        current_re2 = re2.q0
        
        for char in test:
            next_state_re1 = re1.d(current_re1, char) 
            next_state_re2 = re2.d(current_re2, char)
            
            # Check if either leads to a trap state
            is_trap_re1 = next_state_re1 in [state for state in re1.Q if state.name == 'T']
            is_trap_re2 = next_state_re2 in [state for state in re2.Q if state.name == 'T']
            
            if not is_trap_re1 and not is_trap_re2:
                enforced += char
                current_re1 = next_state_re1
                current_re2 = next_state_re2
        
        print(f"Input: '{test}' -> Output: '{enforced}'")

    # And the longer test case
    test_long = "abcaabbcaac"
    print(f"\nTesting Intersection (RE1 ∩ RE2) with input '{test_long}':")
    enforced = ''
    current_re1 = re1.q0
    current_re2 = re2.q0

    for char in test_long:
        next_state_re1 = re1.d(current_re1, char) 
        next_state_re2 = re2.d(current_re2, char)
        
        # Check if either leads to a trap state
        is_trap_re1 = next_state_re1 in [state for state in re1.Q if state.name == 'T']
        is_trap_re2 = next_state_re2 in [state for state in re2.Q if state.name == 'T']
        
        if not is_trap_re1 and not is_trap_re2:
            enforced += char
            current_re1 = next_state_re1
            current_re2 = next_state_re2

    print(f"Input: '{test_long}' -> Output: '{enforced}'")
