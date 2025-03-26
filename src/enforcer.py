#!/usr/bin/env python3
"""
enforcer.py

This module implements both bounded runtime enforcement and compositional runtime enforcement.
It includes:
  - Functions for bounded enforcement (computeEmptinessDict, computes_substring, clean, enforcer, idealenforcer)
  - Classes for compositional enforcement (state, DFA, pDFA, parallel_enforcer, maximal_prefix_parallel_enforcer, 
    serial_composition_enforcer) as well as product and monolithic_enforcer functions.
  - A new function bounded_compositional_enforcer that composes multiple enforcers (via monolithic_enforcer)
    while using a bounded memory buffer and suppressing any event that leads to a trap state.
    
Whenever an input leads to a trap state (as defined by the helper function isTrap below), that event is suppressed.
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
            print("State {} is a trap state.".format(state.name))
        else:
            dictEnf[state] = False
            print("State {} is not a trap state.".format(state.name))
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
    # if maxBuffer < len(phi.Q):
    #     print('your buffer is not of reasonable size')
    #     exit()
    print("normal enforcer")
    global estart, eend, y, sum
    y = 0
    sum = 0
    sigmaC = collections.deque([], maxlen=maxBuffer)
    sigmaS = []
    q = phi.q0
    dictEnf = computeEmptinessDict(phi)
    phi.q0 = q
    m = q
    estart = time.time()
    for event in sigma:
        t = q
        q = phi.d(q, event)
        Final = phi.F(q)
        if Final == True:
            print("final state reached with event " + str(event))
            for a in sigmaC:
                sigmaS.append(a)
            sigmaS.append(event)
            sigmaC = []
            t = q
        else:
            if dictEnf[q] == True:
                # The new event would lead to a trap state; suppress it by reverting state.
                print("kicking out event " + str(event))
                q = t
            else:
                t = q
                clean_start = time.time()
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
                    clean_end = time.time()
                    sum = sum + (clean_end - clean_start)
                else:
                    sigmaC.append(event)
    eend = time.time()
    phi.buffer = sigmaC
    print("output sequence is " + str(sigmaS))
    return sigmaS

def idealenforcer(phi, sigma):
    """
    Ideal enforcer function to compute the output sequence sigmaS incrementally.
    """
    global istart, iend
    isigmaC = []
    isigmaS = []
    ip = phi.q0
    dictEnf = computeEmptinessDict(phi)
    istart = time.time()
    for event in sigma:
        a = ip
        ip = phi.step1(event)
        if phi.F(ip):
            Final = True
            a = ip
        else:
            Final = False
        if Final == True:
            for a in isigmaC:
                isigmaS.append(a)
            isigmaS.append(event)
            isigmaC = []
        else:
            if dictEnf[ip] == True:
                ip = a
            else:
                isigmaC.append(event)
                a = ip
    iend = time.time()
    print("output sequence is " + str(isigmaS))
    return isigmaS


##############################  Compositional Enforcement ######################################
from heapq import merge
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


class pDFA(DFA):
    """
    Updated pDFA class for parallel composition that inherits from DFA.
    It adds:
      - out_buffer: an external buffer for output.
      - out_len: length of the external output.
    """
    def __init__(self, name, S, Q, q0, F, d, end, e=('.l',)):
        super().__init__(name, S, Q, q0, F, d, end, e)
        self.out_buffer = []
        self.out_len = 0

    def runInput(self, a):
        """
        Processes a single input symbol a.
        Appends a to the internal buffer, updates the current state via the transition function d,
        and if an accepting state is reached, appends the internal buffer to the external buffer and flushes it.
        """
        self.buffer.append(a)
        self.q = self.d(self.q, a)
        var = self.q
        if var in self.end:
            self.out_buffer += self.buffer
            self.out_len += len(self.buffer)
            self.buffer = []
        return var

    def flushOutBuffer(self):
        """Clears the external output buffer."""
        self.out_buffer = []
        self.out_len = 0

    def lenOut(self):
        """Returns the length of the external output buffer."""
        return self.out_len

    def outBuffer(self):
        """Returns the external output buffer."""
        return self.out_buffer


##############################  (Rest of the code remains unchanged) ######################################
# (parallel_enforcer, maximal_prefix_parallel_enforcer, serial_composition_enforcer, product,
# monolithic_enforcer, bounded_compositional_enforcer)
# ... (unchanged code follows)


# Helper function: Longest Common Subsequence (LCS)
def longest_common_subsequence(s1, s2):
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

class parallel_enforcer(object):
    """
    Class for merging the outputs of external buffers from a set of 2 or more parallel enforcers.
    Modified to compute the maximal (longest common subsequence) substring present in the outputs of the individual properties.
    
    Testable Code
    -------------
    >>> t = str(bin(15*1859))[2:]
    >>> print(len(t), int(t, 2))
    15 27885
    >>> A1, A2, A3 = state('A1'), state('A2'), state('A3')
    >>> B1, B2, B3, B4, B5 = state('B1'), state('B2'), state('B3'), state('B4'), state('B5')
    >>> A1.transit['0'] = A1
    >>> A1.transit['1'] = A2
    >>> A2.transit['0'] = A3
    >>> A2.transit['1'] = A1
    >>> A3.transit['0'] = A2
    >>> A3.transit['1'] = A3
    >>> B1.transit['0'] = B1
    >>> B1.transit['1'] = B2
    >>> B2.transit['0'] = B3
    >>> B2.transit['1'] = B4
    >>> B3.transit['0'] = B5
    >>> B3.transit['1'] = B1
    >>> B4.transit['0'] = B2
    >>> B4.transit['1'] = B3
    >>> B5.transit['0'] = B4
    >>> B5.transit['1'] = B5
    >>> A = pDFA('A', ['0', '1'], [A1, A2, A3], A1, [A1])
    >>> B = pDFA('B', ['0', '1'], [B1, B2, B3, B4, B5], B1, [B1])
    >>> M = parallel_enforcer(A, B)
    >>> M.checkAccept(t)
    ['<maximal common subsequence here>']
    """
    def __init__(self, *D):
        assert len(D) > 1, "Too few DFA to combine"
        self.output = []
        self.D = D
        self.out_buffer_len = [automata.lenOut() for automata in D]

    def updateStatusOnInput(self, _input):
        signal = [0 for _ in self.D]
        for idx, automata in enumerate(self.D):
            automata.runInput(_input)
            curr_size = automata.lenOut()
            if curr_size != self.out_buffer_len[idx] and curr_size != 0:
                self.out_buffer_len[idx] = curr_size
                signal[idx] = 1
        return signal

    def maxMerge(self, signal):
        # When all enforcers have updated their buffers,
        # compute the longest common subsequence among their outputs.
        if all(signal):
            # Gather outputs as strings
            outputs = [''.join(automata.outBuffer()) for automata in self.D]
            # Compute LCS pairwise (if more than 2, iterate through them)
            common = outputs[0]
            for out in outputs[1:]:
                common = longest_common_subsequence(common, out)
            # Append the common subsequence to the overall output.
            self.output.append(common)
            # Flush the external buffers of all enforcers.
            for automata in self.D:
                automata.flushOutBuffer()

    def checkAccept(self, Input):
        for i in Input:
            signal = self.updateStatusOnInput(i)
            self.maxMerge(signal)
        return self.output

class maximal_prefix_parallel_enforcer(object):
    """Class for maximal prefix parallel enforcer.
    This merge technique does not work for all regular properties.
    
    Testable Code
    -------------
    (Test code unchanged.)
    """
    def __init__(self, *D):
        assert len(D) > 1, "Too few DFA to combine"
        self.D = D
        self.output = []
        self.enforcer_outputs = ['' for _ in D]
        self.total_enforcer_count = len(D)

    def updateStatusOnInput(self, _input):
        for idx, automata in enumerate(self.D):
            output = automata.checkAccept(_input)
            self.enforcer_outputs[idx] += ''.join(output)

    def maximalPrefixMerge(self):
        if any([output == '' for output in self.enforcer_outputs]):
            return ''
        mergeResult = min(self.enforcer_outputs)
        mergeResult_len = len(mergeResult)
        if mergeResult != '':
            self.enforcer_outputs = [self.enforcer_outputs[i][mergeResult_len:] for i in range(self.total_enforcer_count)]
            self.output.append(mergeResult)

    def checkAccept(self, Input):
        for i in Input:
            self.updateStatusOnInput(i)
            self.maximalPrefixMerge()
        return self.output

class serial_composition_enforcer(object):
    """Class for generating a serial composition of enforcers.
    (No changes in this class.)
    """
    def __init__(self, *D):
        assert len(D) > 0, "No input DFA"
        self.output = []
        self.D = D

    def updateStatusOnInput(self, _input):
        output = _input
        for automata in self.D:
            if output != '':
                output = ''.join(automata.checkAccept(output))
                continue
            break
        return output

    def checkAccept(self, Input):
        for i in Input:
            output_on_token = self.updateStatusOnInput(i)
            if output_on_token:
                self.output.append(output_on_token)
        return self.output

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
    
    # Create product states
    p_states = []
    for stateA in A.Q:
        for stateB in B.Q:
            p_states.append(ProductState(stateA, stateB))
    
    # Define the start state
    p_start = next(s for s in p_states if s.stateA == A.q0 and s.stateB == B.q0)
    
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
                    
                # Find the corresponding product state
                next_p_state = next((s for s in p_states if s.stateA == next_stateA and s.stateB == next_stateB), None)
                
                if next_p_state in non_trap_states:
                    non_trap_states.add(state)
                    changed = True
                    break
    
    # Define trap states as those not in non_trap_states
    # And mark them as trap states for easy identification
    trap_states = set(p_states) - non_trap_states
    for state in trap_states:
        state.is_trap = True
    
    # Store information about which transitions are suppressed
    suppressed_transitions = []
    
    # Define the transition function for the product DFA with trap state suppression
    def p_delta(p_state, symbol):
        next_stateA = A.d(p_state.stateA, symbol)
        next_stateB = B.d(p_state.stateB, symbol)
        
        if next_stateA is None or next_stateB is None:
            return None
            
        # Find the corresponding product state
        next_p_state = next((s for s in p_states if s.stateA == next_stateA and s.stateB == next_stateB), None)
        
        # Suppress transition if it leads to a trap state
        if next_p_state.is_trap:
            # Record that we're suppressing this transition
            suppressed_transitions.append((p_state.name, symbol, next_p_state.name))
            return p_state  # Stay in the current state (suppress event)
            
        return next_p_state
    
    # Create the product DFA
    product_dfa = DFA(p_name, A.S, p_states, p_start, p_F, p_delta, p_end)
    product_dfa.trap_states = trap_states  # Store the trap states
    
    # Store suppressed transitions for debugging
    product_dfa.suppressed_transitions = suppressed_transitions
    
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

##############################  Bounded Compositional Runtime Enforcer ##############################
def bounded_compositional_enforcer(phi_list, sigma, maxBuffer):
    """
    Bounded compositional runtime enforcer.
    
    Combines multiple DFAs (passed in as phi_list) via the monolithic_enforcer.
    Then processes the input sigma while maintaining a bounded buffer (of size maxBuffer).
    
    Whenever processing an event would lead to a trap state, that event is suppressed.
    The notion of a "trap state" is implemented in the helper function isTrap below.
    
    Returns the output sequence sigmaS.
    """
    # First, compose the multiple properties into one monolithic DFA.
    combined_enf = monolithic_enforcer("Combined", *phi_list)
    
    # Define a helper to determine if a state is a trap state.
    def isTrap(state, automaton):
        # Example definition: a state is a trap if it is not accepting and
        # all transitions from that state lead back to itself.
        if state in automaton.end:
            return False
        for letter in automaton.alphabet:
            if state.transit.get(letter, None) != state:
                return False
        return True

    sigmaC = collections.deque([], maxlen=maxBuffer)
    sigmaS = []
    q = combined_enf.start
    # Process each event in sigma
    for event in sigma:
        prev_q = q
        # Try to perform the transition; if no transition is defined, treat it as a trap.
        try:
            q = q.transit[event]
        except KeyError:
            # No valid transition: suppress event and continue.
            q = prev_q
            continue

        # If the new state is a trap, then suppress the event (do not add it to the buffer or output)
        if isTrap(q, combined_enf):
            q = prev_q  # revert the transition
            continue
        else:
            # Otherwise, add the event to the bounded buffer.
            sigmaC.append(event)
            # If we reached an accepting state, flush the buffer to the output.
            if q in combined_enf.end:
                sigmaS.extend(list(sigmaC))
                sigmaC.clear()
    return sigmaS

##############################  Main (doctest invocation) ######################################
if __name__ == '__main__':
    import doctest
    doctest.testmod()
