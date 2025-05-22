#!/usr/bin/env python3
"""
enforcer.py

This module implements both bounded runtime enforcement and compositional runtime enforcement.
It includes:
  - Functions for bounded enforcement (computeEmptinessDict, computes_substring, clean, enforcer)
  - Functions for compositional enforcement (product, monolithic_enforcer, serial_enforcer)
  - Utility function for longest common subsequence calculation
"""

##############################  Bounded Runtime Enforcement ######################################
import collections
import itertools
from itertools import islice
import time

def computeEmptinessDict(autC):
    """
    Pre-compute the emptiness check for each state in the given automaton.
    Returns a dictionary mapping each state to True if its language is empty (i.e. trap state), else False.
    """
    dictEnf = {}
    for state in autC.Q:
        autC.makeInit(state)
        if autC.isEmpty():
            dictEnf[state] = True
        else:
            dictEnf[state] = False
    return dictEnf

def computes_substring(iterable, n, automata, k):
    """
    Computes a substring of sigmaC by removing the smallest cycle in sigmaC.
    Returns a list [n+1, cleanedBuffer] where cleanedBuffer is the remainder after removing the cycle.
    """
    cleanedBuffer = []
    automata.reset(k)
    p1 = k
    for i in range(len(iterable) - n):
        element = list(islice(iterable[i:], 0, n + 1, 1))
        for j in range(n + 1):
            p2 = automata.step1(element[j])
            if j == 0:
                p3 = p2
        if p2 == p1:
            cleanedBuffer.extend(iterable[i + n + 1:])
            return [n + 1, cleanedBuffer]
        else:
            cleanedBuffer.append(element[0])
        p1 = p3
        automata.makeInit(p3)
        automata.reset(p1)

def clean(sigmaC, phiautomata, maxBuffer, k, event):
    """
    Returns a cleaned sigmaC.
    If a cycle is detected, the cycle is removed and the event is appended.
    """
    yn = None
    for i in range(len(sigmaC)):
        if yn is None:
            yn = computes_substring(list(sigmaC), i, phiautomata, k)
            if i == 0 and yn is None:
                for t in sigmaC:
                    q_q = phiautomata.d(k, t)
                if phiautomata.d(q_q, event) == q_q:
                    return sigmaC
    if yn is not None:
        yn = yn[1:]
        yn = list(itertools.chain(*yn))
        yn.append(event)
        return yn

def enforcer(phi, sigma, maxBuffer):
    """
    Bounded memory enforcer function to compute the output sequence sigmaS incrementally.
    Simplified to eliminate redundant trap state detection logic.
    """
    y = 0
    sum = 0
    sigmaC = collections.deque([], maxlen=maxBuffer)
    sigmaS = []
    q = phi.q0
    
    
    if hasattr(phi, 'dictEnf'):
        dictEnf = phi.dictEnf
    else:
        dictEnf = computeEmptinessDict(phi)
        
    phi.q0 = q
    m = q
    
    if not hasattr(phi, 'suppressed_transitions'):
        phi.suppressed_transitions = {}

    # print("\n=== Enforcer Transition Tracing ===")
    # print(f"Starting state: {q.name}")
    
    # Process each event in the input
    for event_idx, event in enumerate(sigma):
        # Store previous state
        prev_state = q
        
        # print(f"\nEvent {event_idx}: '{event}'")
        # print(f"  From state: {prev_state.name}")
        
        
        should_suppress = False
        # print(phi.suppressed_transitions)
        if hasattr(phi, 'suppressed_transitions') and prev_state in phi.suppressed_transitions:
            should_suppress = event in phi.suppressed_transitions[prev_state]
            # if should_suppress:
            #     print(f"  SUPPRESSED: Event '{event}' is in suppressed_transitions")
        
        
        if not should_suppress:
            next_state = phi.d(q, event)
            # print("prisha")
            # print(f"  To state: {next_state.name}")
            
            
            is_trap = False
            
            
            if next_state in dictEnf:
                is_trap = dictEnf[next_state]
                if is_trap:
                    # print(f"  SUPPRESSED: State in emptiness dictionary")
                    should_suppress = True
            
            
            if not is_trap and hasattr(next_state, 'stateA') and hasattr(next_state, 'stateB'):
                comp_trap = (next_state.stateA.name == 'T' or next_state.stateB.name == 'T')
                if comp_trap:
                    # print(f"  SUPPRESSED: Component trap state detected")
                    should_suppress = True
                    
                    
                    if not hasattr(phi, 'suppressed_transitions'):
                        phi.suppressed_transitions = {}
                    if prev_state not in phi.suppressed_transitions:
                        phi.suppressed_transitions[prev_state] = set()
                    phi.suppressed_transitions[prev_state].add(event)
            
            
            if not should_suppress and hasattr(next_state, 'is_trap') and next_state.is_trap:
                # print(f"  SUPPRESSED: is_trap attribute is True")
                should_suppress = True
                
                
                if not hasattr(phi, 'suppressed_transitions'):
                    phi.suppressed_transitions = {}
                if prev_state not in phi.suppressed_transitions:
                    phi.suppressed_transitions[prev_state] = set()
                phi.suppressed_transitions[prev_state].add(event)
       
        
        if should_suppress:
            continue
            
       
        next_state = phi.d(q, event)
        q = next_state
        # print(f"  ACCEPTED: Moving to state {q.name}")
    
        
        
        if phi.F(q):
            # print(f"  In accepting state - flushing buffer + current event")
            for a in sigmaC:
                sigmaS.append(a)
            sigmaS.append(event)
            # print(f"  Output: {sigmaS}")
            sigmaC.clear()
        else:
            if len(sigmaC) >= maxBuffer:
                # print(f"  Buffer full ({len(sigmaC)}/{maxBuffer}) - cleaning buffer")
                phi.q0 = m
                k = phi.q0
                for t in sigmaS:
                    k = phi.d(k, t)
                y = y + 1
                sigmaC1 = clean(sigmaC, phi, maxBuffer, k, event)
                if sigmaC1 == 100:
                    # print(f"  Buffer cleaning failed")
                    break
                else:
                    # print(f"  Buffer cleaned: {sigmaC} → {sigmaC1}")
                    sigmaC = sigmaC1
            else:
                sigmaC.append(event)
                # print(f"  Added to buffer: {list(sigmaC)}")
    
    # Store the remaining buffer in the automaton
    phi.buffer = sigmaC
    # print("\n=== Enforcement Complete ===")
    # print(f"Final state: {q.name}")
    # print(f"Final buffer: {list(sigmaC)}")
    # print(f"Output sequence: {sigmaS}")
    # print("========================\n")
    
    return sigmaS

##############################  Compositional Enforcement ######################################
import sys
sys.path.append("..")

class state(object):
    """Defines a basic state with a dictionary of transitions from that state."""
    def __init__(self, name):
        self.name = name
        self.transit = dict()

from src.Automaton import DFA as BaseDFA

class DFA(BaseDFA):
    """
    Updated DFA class that uses attributes from Automaton.py.
    In addition to the base functionality, it maintains:
      - buffer: internal buffer for input events.
      - end: a collection of accepting states.
    """
    def __init__(self, name, S, Q, q0, F, d, end, e=('.l',)):
        super().__init__(S, Q, q0, F, d, e)
        self.name = name
        self.end = end
        self.buffer = []

    def runInput(self, sigma, maxBuffer=5):
        """
        Processes the entire input sequence sigma using the bounded enforcer function.
        Returns a tuple (current_state, output_sequence) where output_sequence is the result
        of processing the input through the enforcer.
        """
        output = enforcer(self, list(sigma), maxBuffer)
        return self.q, output

    def checkAccept(self, sigma):
        """
        For testing purposes: process the input sigma with bounded enforcement and return the output sequence.
        """
        return enforcer(self, list(sigma), maxBuffer=5)

    def __flushBuffer(self):
        """Clears the internal buffer."""
        self.buffer = []
"""
def longest_common_subsequence(s1, s2):
    #Computes the longest common subsequence between two strings. Used for parallel compositional enforcement. Args: (s1: First string,s2: Second string). Returns: String representing the longest common subsequence
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    i, j = m, n
    lcs_chars = []
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs_chars.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return ''.join(reversed(lcs_chars))
"""

def product(A, B, p_name):
    """
    Computes the product automaton of two DFAs.
    Returns a DFA which is the product automaton of DFAs A and B with trap state suppression.
    Also tracks which events are accepted vs suppressed.
    """
    assert A.S == B.S, "Alphabets not matching!"
    
    class ProductState:
        def __init__(self, stateA, stateB):
            self.name = f"{stateA.name}_{stateB.name}"
            self.stateA = stateA
            self.stateB = stateB
            
            self.is_trap = False
        
        def __str__(self):
            return self.name
        
        def __repr__(self):
            return self.name
        
        def __eq__(self, other):
            if not isinstance(other, ProductState):
                return False
            return self.name == other.name
        
        def __hash__(self):
            return hash(self.name)
    
    p_states = []
    p_state_map = {}  
    #p_states consists of all the product states
    #p_state_map is a dictionary to map the product states to their component states
    
    for stateA in A.Q:
        for stateB in B.Q:
            prod_state = ProductState(stateA, stateB)
            p_states.append(prod_state)
            # Use a tuple of state names as key for the map
            p_state_map[(stateA.name, stateB.name)] = prod_state
    
    p_start = p_state_map[(A.q0.name, B.q0.name)]
    
    def p_F(p_state):
        return A.F(p_state.stateA) and B.F(p_state.stateB)
    
    p_end = [s for s in p_states if p_F(s)]
    
    non_trap_states = set(p_end)
    changed = True
    while changed:
        changed = False
        for state in p_states:
            if state in non_trap_states:
                continue
            for symbol in A.S:
                next_stateA = A.d(state.stateA, symbol)
                next_stateB = B.d(state.stateB, symbol)
                
                if next_stateA is None or next_stateB is None:
                    continue
                    
                next_p_state = p_state_map.get((next_stateA.name, next_stateB.name))
                
                if next_p_state in non_trap_states:
                    non_trap_states.add(state)
                    changed = True
                    break
    
    trap_states = set(p_states) - non_trap_states
    for state in trap_states:
        state.is_trap = True
    
    suppressed_transitions = {}
    
    def p_delta(p_state, symbol):
        next_stateA = A.d(p_state.stateA, symbol)
        next_stateB = B.d(p_state.stateB, symbol)
        next_p_state = p_state_map.get((next_stateA.name, next_stateB.name))
        
        if next_stateA is None or next_stateB is None:
            return None
        
        is_trap_A = next_stateA.name == 'T'
        is_trap_B = next_stateB.name == 'T'
        
        if is_trap_A or is_trap_B:
            if p_state not in suppressed_transitions:
                suppressed_transitions[p_state] = set()
            suppressed_transitions[p_state].add(symbol)
            # print(f"  SUPPRESSED: Transition to trap state detected ({next_stateA.name}, {next_stateB.name})")
           
        
        return next_p_state
    
    product_dfa = DFA(p_name, A.S, p_states, p_start, p_F, p_delta, p_end)
    
    product_dfa.suppressed_transitions = suppressed_transitions
    
    dictEnf = {}
    for state in p_states:
        if state.is_trap or state.stateA.name == 'T' or state.stateB.name == 'T':
            dictEnf[state] = True
        else:
            dictEnf[state] = False
    
    product_dfa.dictEnf = dictEnf
    product_dfa.trap_states = trap_states
    
    return product_dfa

def monolithic_enforcer(name, *D):
    """
    Generates a monolithic enforcer by composing multiple DFAs via the product construction.
    """
    def combine_properties(name, *D):
        assert len(D) > 1, "Too few DFA to combine"
        combined_enforcer = product(D[0], D[1], name)
        for i in range(2, len(D)):
            combined_enforcer = product(combined_enforcer, D[i], name)
        return combined_enforcer
    return combine_properties(name, *D)

def serial_enforcer(name, *D):
    """
    Implements serial composition of multiple enforcers.
    Properties are enforced sequentially: the output of one enforcer is fed as input to the next.
    
    Args:
        name: Name for the serial composition
        *D: Variable number of DFA objects representing the properties to enforce
        
    Returns:
        A function that performs serial enforcement of the given properties
    """
    def serial_enforcement(sigma, maxBuffer=5):
        assert len(D) > 0, "No DFAs provided for serial enforcement"
        
        current_output = list(sigma)
        
        individual_outputs = {}
        
        for i, dfa in enumerate(D):
            dfa_name = getattr(dfa, 'name', f"Property_{i}")
            current_output = enforcer(dfa, current_output, maxBuffer)
            individual_outputs[dfa_name] = current_output.copy()
            # print(f"After enforcing {dfa_name}: {''.join(current_output)}")
        
        return current_output, individual_outputs
    
    return serial_enforcement

##############################  Main (doctest invocation) ######################################
if __name__ == '__main__':
    import doctest
    doctest.testmod()
