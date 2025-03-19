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
    if maxBuffer < len(phi.Q):
        print('your buffer is not of reasonable size')
        exit()
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
            for a in sigmaC:
                sigmaS.append(a)
            sigmaS.append(event)
            sigmaC = []
            t = q
        else:
            if dictEnf[q] == True:
                # The new event would lead to a trap state; suppress it by reverting state.
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
    print("output sequence is " + str(sigmaS))
    
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


##############################  Compositional Enforcement ######################################
from heapq import merge
import sys
sys.path.append("..")

class state(object):
    """Defines a basic state with a dictionary of transitions from that state."""
    def __init__(self, name):
        self.name = name
        self.transit = dict()

class DFA(object):
    """Class for enforcing a single property.

    Testable Code
    -------------
    >>> a = state('a')
    >>> b = state('b')
    >>> a.transit['0'] = b
    >>> b.transit['0'] = a
    >>> b.transit['1'] = b
    >>> a.transit['1'] = a
    >>> D = DFA('D', ['0', '1'], [a, b], a, [a])
    >>> input1 = '001010010'
    >>> D.checkAccept(input1)
    ['00', '1', '010', '010']
    >>> input2 = '000'
    >>> D.checkAccept(input2)
    ['00']
    >>> input3 = '0100110'
    >>> D.checkAccept(input3)
    ['00', '1', '00', '1', '1']
    >>> input4 = '111'
    >>> D.checkAccept(input4)
    []
    >>> input5 = '0010'
    >>> D.checkAccept(input5)
    ['01110', '010']
    """
    def __init__(self, name, alphabet, states=None, start=None, end=None):
        self.name = name
        self.states = states
        self.alphabet = alphabet
        self.start = start
        self.end = end
        self.curr_state = self.start
        self.buffer = []

    def runInput(self, _input):
        self.buffer.append(_input)
        self.curr_state = self.curr_state.transit[_input]
        var = self.curr_state
        if var in self.end:
            self.buffer = []  # Flush Internal Buffer
        return var

    def checkAccept(self, _input):
        index = []
        output = []
        buffer_on_flush = ''
        if self.buffer:
            buffer_on_flush = ''.join(self.buffer)
        for idx, i in enumerate(_input):
            State = self.runInput(i)
            if State in self.end:
                index.append(idx)
        if index:
            output.append(buffer_on_flush + _input[: index[0] + 1])
            for i in range(len(index) - 1):
                output.append(_input[index[i] + 1: index[i + 1] + 1])
        return output

    def __flushBuffer(self):
        self.buffer = []

class pDFA(DFA):
    """Class for enforcing a single property through parallel composition.

    Testable Code
    -------------
    >>> a = state('a')
    >>> b = state('b')
    >>> a.transit['0'] = b
    >>> b.transit['0'] = a
    >>> b.transit['1'] = b
    >>> a.transit['1'] = a
    >>> D = pDFA('D', ['0', '1'], [a, b], a, [a])
    >>> input1 = '001010010'
    >>> D.checkAccept(input1)
    ['00', '1', '010', '010']
    >>> D.outBuffer()
    ['0', '0', '1', '0', '1', '0', '0', '1', '0']
    >>> input2 = '000'
    >>> D.checkAccept(input2)
    ['00']
    >>> D.outBuffer()
    ['0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '0']
    >>> input3 = '010011'
    >>> D.checkAccept(input3)
    ['00', '1', '00', '1', '1']
    >>> D.outBuffer()
    ['0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '1', '1']
    """
    def __init__(self, name, alphabet, states=None, start=None, end=None):
        super().__init__(name, alphabet, states, start, end)
        self.out_buffer = []
        self.out_len = 0

    def runInput(self, _input):
        self.buffer.append(_input)
        self.curr_state = self.curr_state.transit[_input]
        var = self.curr_state
        if var in self.end:
            self.out_buffer += self.buffer  # Add to External Buffer
            self.out_len += len(self.buffer)
            self.buffer = []  # Flush Internal Buffer
        return var

    def flushOutBuffer(self):
        self.out_buffer = []
        self.out_len = 0

    def lenOut(self):
        return self.out_len

    def outBuffer(self):
        return self.out_buffer

class parallel_enforcer(object):
    """Class for maximally merging the contents of external buffers of a set of 2 or more parallel enforcers.
    This merge technique works for all regular properties.
    
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
    ['110110011', '101101']
    """
    def __init__(self, *D):
        assert len(D) > 1, "Too few DFA to combine"
        self.output = []
        self.D = D
        self.out_buffer_len = [automata.lenOut() for automata in D]

    def updateStatusOnInput(self, _input):
        signal = [0 for automata in self.D]
        for idx, automata in enumerate(self.D):
            automata.runInput(_input)
            curr_size = automata.lenOut()
            if curr_size != self.out_buffer_len[idx] and curr_size != 0:
                self.out_buffer_len[idx] = curr_size
                signal[idx] = 1
        return signal

    def maxMerge(self, signal):
        if all(signal) == True:
            enforced = self.D[0].outBuffer()
            for automata in self.D:
                automata.flushOutBuffer()
            self.output.append(''.join(enforced))

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
    >>> M = maximal_prefix_parallel_enforcer(A, B)
    >>> M.checkAccept(t)
    ['110110011', '101101']
    >>> t = 'abc'
    >>> A1, A2, A3 = state('A1'), state('A2'), state('A3')
    >>> B1, B2, B3, B4, B5 = state('B1'), state('B2'), state('B3'), state('B4'), state('B5')
    >>> A1.transit['a'] = A2
    >>> A1.transit['b'] = A3
    >>> A1.transit['c'] = A3
    >>> A2.transit['a'] = A3
    >>> A2.transit['b'] = A1
    >>> A2.transit['c'] = A1
    >>> A3.transit['a'] = A3
    >>> A3.transit['b'] = A3
    >>> A3.transit['c'] = A3
    >>> B1.transit['a'] = B2
    >>> B1.transit['b'] = B5
    >>> B1.transit['c'] = B5
    >>> B2.transit['a'] = B5
    >>> B2.transit['b'] = B3
    >>> B2.transit['c'] = B5
    >>> B3.transit['a'] = B5
    >>> B3.transit['b'] = B5
    >>> B3.transit['c'] = B4
    >>> B4.transit['a'] = B4
    >>> B4.transit['b'] = B4
    >>> B4.transit['c'] = B4
    >>> B5.transit['a'] = B5
    >>> B5.transit['b'] = B5
    >>> B5.transit['c'] = B5
    >>> A = pDFA('A', ['a', 'b', 'c'], [A1, A2, A3], A1, [A1])
    >>> B = pDFA('B', ['a', 'b', 'c'], [B1, B2, B3, B4, B5], B1, [B4])
    >>> M = maximal_prefix_parallel_enforcer(A, B)
    >>> M.checkAccept(t)
    ['ab']
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
    >>> A = DFA('A', ['0', '1'], [A1, A2, A3], A1, [A1])
    >>> B = DFA('B', ['0', '1'], [B1, B2, B3, B4, B5], B1, [B1])
    >>> M = serial_composition_enforcer(A, B)
    >>> M.checkAccept(t)
    ['110110011', '101101']
    >>> N = serial_composition_enforcer(B, A)
    >>> N.checkAccept(t)
    ['110110011', '101101']
    >>> t = 'abac'
    >>> A1, A2, A3 = state('A1'), state('A2'), state('A3')
    >>> B1, B2, B3, B4, B5 = state('B1'), state('B2'), state('B3'), state('B4'), state('B5')
    >>> A1.transit['a'] = A2
    >>> A1.transit['b'] = A3
    >>> A1.transit['c'] = A3
    >>> A2.transit['a'] = A3
    >>> A2.transit['b'] = A1
    >>> A2.transit['c'] = A1
    >>> A3.transit['a'] = A3
    >>> A3.transit['b'] = A3
    >>> A3.transit['c'] = A3
    >>> B1.transit['a'] = B2
    >>> B1.transit['b'] = B5
    >>> B1.transit['c'] = B5
    >>> B2.transit['a'] = B5
    >>> B2.transit['b'] = B3
    >>> B2.transit['c'] = B5
    >>> B3.transit['a'] = B4
    >>> B3.transit['b'] = B5
    >>> B3.transit['c'] = B5
    >>> B4.transit['a'] = B5
    >>> B4.transit['b'] = B5
    >>> B4.transit['c'] = B1
    >>> B5.transit['a'] = B5
    >>> B5.transit['b'] = B5
    >>> B5.transit['c'] = B5
    >>> A = DFA('A', ['a', 'b', 'c'], [A1, A2, A3], A1, [A1])
    >>> B = DFA('B', ['a', 'b', 'c'], [B1, B2, B3, B4, B5], B1, [B4])
    >>> M = serial_composition_enforcer(A, B)
    >>> M.checkAccept(t)
    ['aba']
    >>> A._DFA__flushBuffer() # Only for independent testing
    >>> B._DFA__flushBuffer() # Only for independent testing
    >>> N = serial_composition_enforcer(B, A)
    >>> N.checkAccept(t)
    ['ab']
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
    Returns a DFA which is the product automaton of DFAs A and B.
    
    Testable Code
    -------------
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
    >>> A = DFA('A', ['0', '1'], [A1, A2, A3], A1, [A1])
    >>> B = DFA('B', ['0', '1'], [B1, B2, B3, B4, B5], B1, [B1])
    >>> print(A.start.transit['1'].name, B.start.name)
    A2 B1
    >>> C = product(A, B, 'C')
    >>> C.alphabet
    ['0', '1']
    >>> C.start.name
    'A1_B1'
    >>> for s in C.end:
    ...     print(s.name)
    A1_B1
    >>> Input = str(bin(15*1859))[2:]
    >>> C.checkAccept(Input)
    ['110110011', '101101']
    """
    class state(object):
        def __init__(self, name):
            self.name = name
            self.transit = dict()
    assert A.alphabet == B.alphabet, "Alphabets not matching!"
    p_states = []
    p_start = None
    p_end = []
    p_var = dict()

    # Create states for Product Automaton
    for state_A in A.states:
        for state_B in B.states:
            Name = state_A.name + '_' + state_B.name
            p_var[Name] = state(Name)

    # Add transition rules for Product Automaton
    for state_A in A.states:
        for state_B in B.states:
            Name = state_A.name + '_' + state_B.name
            for letter in A.alphabet:
                next_state = state_A.transit[letter].name + '_' + state_B.transit[letter].name
                p_var[Name].transit[letter] = p_var[next_state]

    # Add states of Product Automaton to list
    for state_name in p_var:
        p_states.append(p_var[state_name])

    # Add start state of Product Automaton
    p_start = p_var[A.start.name + '_' + B.start.name]

    # Add end states of Product Automaton to list
    for end_state_A in A.end:
        for end_state_B in B.end:
            p_end.append(p_var[end_state_A.name + '_' + end_state_B.name])

    return DFA(p_name, A.alphabet, p_states, p_start, p_end)

def monolithic_enforcer(name, *D):
    """
    Generates a monolithic enforcer by composing multiple DFAs via the product construction.
    
    Testable Code
    -------------
    >>> alpha = ['0', '1']
    >>> A1, A2, A3 = state('A1'), state('A2'), state('A3')
    >>> A1.transit['0'] = A1
    >>> A1.transit['1'] = A2
    >>> A2.transit['0'] = A3
    >>> A2.transit['1'] = A1
    >>> A3.transit['0'] = A2
    >>> A3.transit['1'] = A3
    >>> A = DFA('A', alpha, [A1, A2, A3], A1, [A1])
    >>> 
    >>> B1, B2, B3, B4, B5 = state('B1'), state('B2'), state('B3'), state('B4'), state('B5')
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
    >>> B = DFA('B', alpha, [B1, B2, B3, B4, B5], B1, [B1])
    >>> 
    >>> C1, C2, C3, C4, C5, C6, C7 = state('C1'), state('C2'), state('C3'), state('C4'), state('C5'), state('C6'), state('C7')
    >>> C1.transit['0'] = C1
    >>> C1.transit['1'] = C2
    >>> C2.transit['0'] = C3
    >>> C2.transit['1'] = C4
    >>> C3.transit['0'] = C5
    >>> C3.transit['1'] = C6
    >>> C4.transit['0'] = C7
    >>> C4.transit['1'] = C1
    >>> C5.transit['0'] = C2
    >>> C5.transit['1'] = C3
    >>> C6.transit['0'] = C4
    >>> C6.transit['1'] = C5
    >>> C7.transit['0'] = C6
    >>> C7.transit['1'] = C7
    >>> C = DFA('C', alpha, [C1, C2, C3, C4, C5, C6, C7], C1, [C1])
    >>> 
    >>> enf_property = monolithic_enforcer("Mono", A, B, C)
    >>> Input = str(bin(105*1859))[2:]
    >>> print(Input)
    101111101001111011
    >>> accept = enf_property.checkAccept(Input)
    >>> print(accept)
    ['101111101001111011']
    """
    def combine_properties(name, *D):
        assert len(D) > 1, "Too few DFA to combine"
        combined_enforcer = product(D[0], D[1], name)
        for i in range(2, len(D)):
            combined_enforcer = product(combined_enforcer, D[i], name)
        return combined_enforcer
    return combine_properties(name, *D)

##############################  New: Bounded Compositional Runtime Enforcer ##############################
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
