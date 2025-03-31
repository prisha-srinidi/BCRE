#!/usr/bin/env python3
"""
enforcer.py

This module implements both bounded runtime enforcement and compositional runtime enforcement.
It includes:
  - Functions for bounded enforcement (computeEmptinessDict, computes_substring, clean, enforcer)
  - Functions for compositional enforcement (product, monolithic_enforcer)
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
    (Trap suppression is implemented by not updating q when the next state is "empty".)
    """
    global estart, eend, y, sum
    y = 0
    sum = 0
    sigmaC = collections.deque([], maxlen=maxBuffer)
    sigmaS = []
    q = phi.q0
    
    # Use pre-computed dictEnf if available, otherwise compute it
    if hasattr(phi, 'dictEnf'):
        dictEnf = phi.dictEnf
    else:
        dictEnf = computeEmptinessDict(phi)
        
    phi.q0 = q
    m = q
    
    # Process each event in the input
    for event in sigma:
        # Store previous state
        prev_state = q
        
        # Check if this transition should be suppressed
        should_suppress = False
        if hasattr(phi, 'suppressed_transitions') and prev_state in phi.suppressed_transitions:
            if event in phi.suppressed_transitions[prev_state]:
                should_suppress = True
        
        if should_suppress:
            # Skip this event entirely
            continue
        
        # Get next state from transition function
        next_state = phi.d(q, event)
        
        # Check if next_state is a trap state using dictEnf
        is_trap = dictEnf.get(next_state, False)
        
        # If not in dictEnf, check if any component is a trap state (for product states)
        if not is_trap and hasattr(next_state, 'stateA') and hasattr(next_state, 'stateB'):
            is_trap = (next_state.stateA.name == 'T' or next_state.stateB.name == 'T')
        
        # Check if the state itself is marked as a trap
        if not is_trap and hasattr(next_state, 'is_trap'):
            is_trap = next_state.is_trap
        
        # If it's a trap state, suppress this event
        if is_trap:
            # Skip this event entirely (don't update state or add to buffer)
            continue
        
        # Not a trap state, update current state
        q = next_state
        
        # If in accepting state, output buffer contents plus current event
        if phi.F(q):
            for a in sigmaC:
                sigmaS.append(a)
            sigmaS.append(event)
            sigmaC.clear()
        else:
            # Not in accepting state, add to buffer (with overflow handling)
            if len(sigmaC) >= maxBuffer:
                phi.q0 = m
                k = phi.q0
                for t in sigmaS:
                    k = phi.d(k, t)
                y = y + 1
                sigmaC1 = clean(sigmaC, phi, maxBuffer, k, event)
                if sigmaC1 == 100:
                    break
                else:
                    sigmaC = sigmaC1
            else:
                sigmaC.append(event)
    
    # Store the remaining buffer in the automaton
    phi.buffer = sigmaC
    # Print for debugging
    print("output sequence is " + str(sigmaS))
    return sigmaS

##############################  Compositional Enforcement ######################################
import sys
sys.path.append("..")

class state(object):
    """Defines a basic state with a dictionary of transitions from that state."""
    def __init__(self, name):
        self.name = name
        self.transit = dict()

# Import the base DFA from Automaton.py to ensure we have methods like isEmpty() available.
from src.Automaton import DFA as BaseDFA

class DFA(BaseDFA):
    """
    Updated DFA class that uses attributes from Automaton.py.
    In addition to the base functionality, it maintains:
      - buffer: internal buffer for input events.
      - end: a collection of accepting states.
    """
    def __init__(self, name, S, Q, q0, F, d, end, e=('.l',)):
        # Initialize the base DFA with the given attributes.
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

# Helper function: Longest Common Subsequence (LCS)
def longest_common_subsequence(s1, s2):
    """
    Computes the longest common subsequence between two strings.
    Used for parallel compositional enforcement.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        String representing the longest common subsequence
    """
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    # Reconstruct LCS from dp table
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
            
            # Add a flag to easily identify if this is a trap state
            self.is_trap = False
        
        def __str__(self):
            return self.name
        
        def __repr__(self):
            return self.name
        
        # Define equality based on the state name
        def __eq__(self, other):
            if not isinstance(other, ProductState):
                return False
            return self.name == other.name
        
        # Use the state name for hashing
        def __hash__(self):
            return hash(self.name)
    
    # Create product states
    p_states = []
    p_state_map = {}  # Map to quickly look up states by component states
    
    for stateA in A.Q:
        for stateB in B.Q:
            prod_state = ProductState(stateA, stateB)
            p_states.append(prod_state)
            # Use a tuple of state names as key for the map
            p_state_map[(stateA.name, stateB.name)] = prod_state
    
    # Define the start state
    p_start = p_state_map[(A.q0.name, B.q0.name)]
    
    # Define the acceptance function for the product DFA
    def p_F(p_state):
        return A.F(p_state.stateA) and B.F(p_state.stateB)
    
    # Define end states
    p_end = [s for s in p_states if p_F(s)]
    
    # Compute states that can reach accepting states (non-trap states)
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
                    
                # Find the corresponding product state using the map
                next_p_state = p_state_map.get((next_stateA.name, next_stateB.name))
                
                if next_p_state in non_trap_states:
                    non_trap_states.add(state)
                    changed = True
                    break
    
    # Define trap states as those not in non_trap_states
    # And mark them as trap states for easy identification
    trap_states = set(p_states) - non_trap_states
    for state in trap_states:
        state.is_trap = True
    
    # Keep track of suppressed transitions for each state
    suppressed_transitions = {}
    
    # Define the transition function for the product DFA
    def p_delta(p_state, symbol):
        next_stateA = A.d(p_state.stateA, symbol)
        next_stateB = B.d(p_state.stateB, symbol)
        
        if next_stateA is None or next_stateB is None:
            return None
        
        # Get the corresponding product state
        next_p_state = p_state_map.get((next_stateA.name, next_stateB.name))
        
        # Check if either component leads to a trap state
        is_trap_A = next_stateA.name == 'T'
        is_trap_B = next_stateB.name == 'T'
        
        # If either leads to a trap state or the product state is marked as a trap,
        # suppress the transition by returning the current state
        if is_trap_A or is_trap_B or next_p_state.is_trap:
            # Mark this as a suppressed transition
            if p_state not in suppressed_transitions:
                suppressed_transitions[p_state] = set()
            suppressed_transitions[p_state].add(symbol)
            
            return p_state  # Stay in current state (suppress the event)
        
        return next_p_state
    
    # Create the product DFA
    product_dfa = DFA(p_name, A.S, p_states, p_start, p_F, p_delta, p_end)
    
    # Store the suppressed transitions in the product DFA
    product_dfa.suppressed_transitions = suppressed_transitions
    
    # Pre-compute and store the emptiness dictionary for the product DFA
    dictEnf = {}
    for state in p_states:
        if state.is_trap or state.stateA.name == 'T' or state.stateB.name == 'T':
            dictEnf[state] = True
        else:
            dictEnf[state] = False
    
    # Store the pre-computed emptiness dictionary and trap states
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

##############################  Main (doctest invocation) ######################################
if __name__ == '__main__':
    import doctest
    doctest.testmod()
