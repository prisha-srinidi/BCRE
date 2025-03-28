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

from src.enforcer import state, DFA, monolithic_enforcer, enforcer

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
    
    # Dictionary to store enforced outputs for each property
    cs3_outputs = {}
    cs4_outputs = {}
    
    print("\nCS3 property (a,*,b,*):")
    print("-------------------------")
    for test in cs_test_strings:
        # Use the enforcer function for CS3
        enforced_output = enforcer(cs3, list(test), 2)  # Use buffer size 5
        enforced = ''.join(enforced_output)
        print(f"Input: '{test}' -> Output: '{enforced}'")
        cs3_outputs[test] = enforced
    
    print("\nCS4 property (a,*,b,c,*):")
    print("--------------------------")
    for test in cs_test_strings:
        # Use the enforcer function for CS4
        enforced_output = enforcer(cs4, list(test), 2)  # Use buffer size 5
        enforced = ''.join(enforced_output)
        print(f"Input: '{test}' -> Output: '{enforced}'")
        cs4_outputs[test] = enforced
    
    # Testing CS3 ∩ CS4 intersection with manual parallel composition
    print("\nTesting Intersection (CS3 ∩ CS4) via Manual Parallel Composition:")
    print("---------------------------------------------------------------")
    for test in cs_test_strings:
        # Manual computation of intersection output with proper trap state detection
        enforced = ''
        current_cs3 = cs3.q0
        current_cs4 = cs4.q0
        
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
    
    # Define a function to find the longest common subsequence (maximal substring)
    def longest_common_subsequence(s1, s2):
        """
        Computes the longest common subsequence between two strings.
        This represents the maximal substring function mentioned in the paper.
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Build the LCS table
        for i in range(m):
            for j in range(n):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        
        # Reconstruct the LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                lcs.append(s1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        
        # Return the reversed LCS (as we built it backwards)
        return ''.join(reversed(lcs))
    
    # Testing CS3 ∩ CS4 intersection using the maximal substring approach
    print("\nTesting Intersection (CS3 ∩ CS4) via Maximal Substring:")
    print("------------------------------------------------------")
    for test in cs_test_strings:
        # Get the enforced outputs from both properties
        output_cs3 = cs3_outputs[test]
        output_cs4 = cs4_outputs[test]
        
        # Compute the longest common subsequence (maximal substring)
        maximal_substring = longest_common_subsequence(output_cs3, output_cs4)
        
        print(f"Input: '{test}'")
        print(f"  CS3 Output: '{output_cs3}'")
        print(f"  CS4 Output: '{output_cs4}'")
        print(f"  Maximal Substring: '{maximal_substring}'")
    
    # Testing CS3 ∩ CS4 intersection using the monolithic enforcer
    print("\nTesting Intersection (CS3 ∩ CS4) via Monolithic Enforcer:")
    print("---------------------------------------------------------")
    from src.enforcer import enforcer  # Import the enforcer function

    # Create the monolithic enforcer by composing CS3 and CS4
    cs_intersection = monolithic_enforcer("CS3_CS4", CS3(), CS4())

    for test in cs_test_strings:
        # Convert input string to list of individual events as required by enforcer function
        input_events = list(test)
        
        # Use the enforcer function from enforcer.py with appropriate buffer size
        max_buffer = 3  # Adjust buffer size as needed
        enforced_output = enforcer(cs_intersection, input_events, max_buffer)
        
        # Join the output list into a string for display
        enforced = ''.join(enforced_output)
        
        print(f"Input: '{test}' -> Output: '{enforced}'")
    
    # Testing CS3 ∩ CS4 intersection using the monolithic enforcer with bounded enforcement
    print("\nTesting Intersection (CS3 ∩ CS4) via Monolithic Enforcer with Bounded Enforcement:")
    print("-------------------------------------------------------------------------------")
    from src.enforcer import enforcer  # Import the bounded enforcer function

    cs_intersection = monolithic_enforcer("CS3_CS4", CS3(), CS4())

    # Use the bounded enforcer function for proper buffering semantics
    for test in cs_test_strings:
        # Create a maximal buffer size - can be adjusted based on property needs
        max_buffer = 3
        
        # Convert input string to list of individual events as required by enforcer function
        input_events = list(test)
        
        # Use the enforcer function from enforcer.py which implements proper buffering
        enforced_output = enforcer(cs_intersection, input_events, max_buffer)
        
        # Join the output list into a string for display
        enforced = ''.join(enforced_output)
        
        print(f"Input: '{test}' -> Output: '{enforced}'")

    re_progressive_strings = [
        "a",    # First character
        "ac",   # First two characters
        "acc",  # First three characters
        "acca", # First four characters
        "accab" # Full test string
    ]
    # Similarly for the RE1 ∩ RE2 intersection
    print("\nTesting Intersection (RE1 ∩ RE2) via Monolithic Enforcer with Bounded Enforcement:")
    print("-------------------------------------------------------------------------------")
    re_intersection = monolithic_enforcer("RE1_RE2", RE1(), RE2())

    for test in re_progressive_strings:
        max_buffer = 10
        input_events = list(test)
        enforced_output = enforcer(re_intersection, input_events, max_buffer)
        enforced = ''.join(enforced_output)
        print(f"Input: '{test}' -> Output: '{enforced}'")

    # And for the additional test cases
    print("\nAdditional Test Cases for Intersection (RE1 ∩ RE2) via Monolithic Enforcer with Bounded Enforcement:")
    print("--------------------------------------------------------------------------------------------")
    # Progressive test strings for 'accab' showing each step
    re_test_strings = [
        "a",       
        "ac",      
        "acc",
        "acca",
        "accab"
    ]
    for test in re_progressive_strings:
        max_buffer = 10
        input_events = list(test)
        enforced_output = enforcer(re_intersection, input_events, max_buffer)
        enforced = ''.join(enforced_output)
        print(f"Input: '{test}' -> Output: '{enforced}'")

    # And for the longer test case
    print("\nTesting Intersection (RE1 ∩ RE2) with combined input string via Monolithic Enforcer:")
    print("----------------------------------------------------------------------------")
    # Join all the progressive strings into a single string
    combined_input = ''.join(re_progressive_strings)
    print(f"Combined input: '{combined_input}'")

    # Convert to list of characters
    input_events = list(combined_input)

    # Use the enforcer function
    max_buffer = 10
    enforced_output = enforcer(re_intersection, input_events, max_buffer)

    # Join the output list into a string for display
    enforced = ''.join(enforced_output)

    print(f"Monolithic Enforcer Output: '{enforced}'")
    
    # Testing the new RE1 and RE2 automata
    re1 = RE1()
    re2 = RE2()
    
    # Store outputs for RE1 and RE2
    re1_outputs = {}
    re2_outputs = {}
    
    
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
        re1_outputs[test] = enforced
    
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
        re2_outputs[test] = enforced
    
    # Test RE1 ∩ RE2 intersection via manual parallel composition
    print("\nTesting Intersection (RE1 ∩ RE2) via Manual Parallel Composition:")
    print("----------------------------------------------------------------")
    for test in re_progressive_strings:
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
    
    # Testing RE1 ∩ RE2 intersection using the maximal substring approach
    print("\nTesting Intersection (RE1 ∩ RE2) via Maximal Substring:")
    print("------------------------------------------------------")
    for test in re_progressive_strings:
        # Get the enforced outputs from both properties
        output_re1 = re1_outputs[test]
        output_re2 = re2_outputs[test]
        
        # Compute the longest common subsequence (maximal substring)
        maximal_substring = longest_common_subsequence(output_re1, output_re2)
        
        print(f"Input: '{test}'")
        print(f"  RE1 Output: '{output_re1}'")
        print(f"  RE2 Output: '{output_re2}'")
        print(f"  Maximal Substring: '{maximal_substring}'")
    
    # Testing RE1 ∩ RE2 intersection using the monolithic enforcer
    print("\nTesting Intersection (RE1 ∩ RE2) via Monolithic Enforcer:")
    print("---------------------------------------------------------")
    # Create the monolithic enforcer for RE1 and RE2
    re_intersection = monolithic_enforcer("RE1_RE2", RE1(), RE2())

    for test in re_progressive_strings:
        # Convert input string to list of individual events
        input_events = list(test)
        
        # Use the enforcer function from enforcer.py
        max_buffer = 3
        enforced_output = enforcer(re_intersection, input_events, max_buffer)
        
        # Join the output list into a string for display
        enforced = ''.join(enforced_output)
        
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
    
    # Store outputs for additional test cases
    re1_additional_outputs = {}
    re2_additional_outputs = {}
    
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
        re1_additional_outputs[test] = enforced
    
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
        re2_additional_outputs[test] = enforced
    
    # Test additional cases with manual parallel composition
    print("\nAdditional Test Cases for Intersection (RE1 ∩ RE2) via Manual Parallel Composition:")
    print("-------------------------------------------------------------------------------")
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
    
    # Test additional cases with maximal substring approach
    print("\nAdditional Test Cases for Intersection (RE1 ∩ RE2) via Maximal Substring:")
    print("-----------------------------------------------------------------------")
    for test in re_test_strings:
        # Get the enforced outputs from both properties
        output_re1 = re1_additional_outputs[test]
        output_re2 = re2_additional_outputs[test]
        
        # Compute the longest common subsequence (maximal substring)
        maximal_substring = longest_common_subsequence(output_re1, output_re2)
        
        print(f"Input: '{test}'")
        print(f"  RE1 Output: '{output_re1}'")
        print(f"  RE2 Output: '{output_re2}'")
        print(f"  Maximal Substring: '{maximal_substring}'")
    
    # Test additional cases with monolithic enforcer
    print("\nAdditional Test Cases for Intersection (RE1 ∩ RE2) via Monolithic Enforcer:")
    print("--------------------------------------------------------------------------")
    for test in re_test_strings:
        # Convert input string to list of individual events
        input_events = list(test)
        
        # Use the enforcer function from enforcer.py
        max_buffer = 10
        enforced_output = enforcer(re_intersection, input_events, max_buffer)
        
        # Join the output list into a string for display
        enforced = ''.join(enforced_output)
        
        print(f"Input: '{test}' -> Output: '{enforced}'")

    # Test the longer string with monolithic enforcer
    test_long="abccaa"
    print(f"\nTesting Intersection (RE1 ∩ RE2) with input '{test_long}' via Monolithic Enforcer:")
    # Convert input string to list of individual events
    input_events = list(test_long)
        
    # Use the enforcer function from enforcer.py
    max_buffer = 10
    enforced_output = enforcer(re_intersection, input_events, max_buffer)

    # Join the output list into a string for display
    enforced = ''.join(enforced_output)

    print(f"Monolithic Enforcer Output: '{enforced}'")
    
    # Test a longer challenging sequence for enforcement
    test_long = "abcaabbcaac"
    
    # Process test_long with individual enforcers
    print(f"\nTesting input '{test_long}' with individual enforcers:")
    
    # RE1 enforcement
    enforced_re1 = ''
    current = re1.q0
    for char in test_long:
        next_state = re1.d(current, char)
        if next_state in [state for state in re1.Q if state.name == 'T']:
            # Suppress the event that leads to trap state
            continue
        else:
            enforced_re1 += char
            current = next_state
    print(f"RE1 Output: '{enforced_re1}'")
    
    # RE2 enforcement
    enforced_re2 = ''
    current = re2.q0
    for char in test_long:
        next_state = re2.d(current, char)
        if next_state in [state for state in re2.Q if state.name == 'T']:
            # Suppress the event that leads to trap state
            continue
        else:
            enforced_re2 += char
            current = next_state
    print(f"RE2 Output: '{enforced_re2}'")
    
    # Test manual parallel composition
    print(f"\nTesting Intersection (RE1 ∩ RE2) with input '{test_long}' via Manual Parallel Composition:")
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
    
    print(f"Manual Parallel Output: '{enforced}'")
    
    # Test maximal substring approach
    print(f"\nTesting Intersection (RE1 ∩ RE2) with input '{test_long}' via Maximal Substring:")
    maximal_substring = longest_common_subsequence(enforced_re1, enforced_re2)
    print(f"Maximal Substring: '{maximal_substring}'")
    
    # Test monolithic enforcer
    print(f"\nTesting Intersection (RE1 ∩ RE2) with input '{test_long}' via Monolithic Enforcer:")
    enforced = ''
    current = re_intersection.q0
    for char in test_long:
        # Store previous state to detect if transition was suppressed
        prev_state = current
        next_state = re_intersection.d(current, char)
        
        # Check if this is a trap state transition by examining the state name
        is_trap = False
        if next_state:
            state_name_parts = next_state.name.split('_')
            # Check if any component is a trap state
            is_trap = 'T' in state_name_parts
        
        # If not a trap state and state changed, accept the event
        if not is_trap and next_state != prev_state:
            enforced += char
            current = next_state
    
    print(f"Monolithic Enforcer Output: '{enforced}'")

    # And for the longer test case - properly concatenate all strings together
    print("\nTesting Intersection (RE1 ∩ RE2) with combined input string via Monolithic Enforcer:")
    print("----------------------------------------------------------------------------")
    # Join all the progressive strings into a single string
    combined_input = ''.join(re_progressive_strings)
    print(f"Combined input: '{combined_input}'")

    # Convert to list of characters
    input_events = list(combined_input)

    # Use the enforcer function
    max_buffer = 10
    enforced_output = enforcer(re_intersection, input_events, max_buffer)

    # Join the output list into a string for display
    enforced = ''.join(enforced_output)

    print(f"Monolithic Enforcer Output: '{enforced}'")
