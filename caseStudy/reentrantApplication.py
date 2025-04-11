#!/usr/bin/env python3
"""
reentrantApplication.py

This file implements runtime enforcement for several reentrant application properties:

1. No Reentrant Call in a Critical Section
   Regular Expression: (o)*(e(o)*x(o)*)*
   
2. Eventual Completion of Interrupt Handling
   Regular Expression: (o)*(i(o)*j(o)*)*
   
3. Bounded Depth of Reentrant Calls (Max Depth = 2)
   Regular Expression: (o)*(c(o)*(c(o)*r(o)*)?r(o)*)*
   
4. No Premature Return
   Regular Expression: (o)*(c(o)*r(o)*)*
   
5. Non-Reentrant Locking Protocol
   Regular Expression: (o)*(l(o)*u(o)*)*

Each property is represented as a DFA and tested with various input sequences,
using both bounded runtime enforcement and compositional enforcement.
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.enforcer import state, DFA, enforcer, monolithic_enforcer

def build_and_test_property(name, alphabet, states, transition_map, start_state, accept_states, test_inputs, max_buffer=20):
    """Helper function to build a DFA and test it with various inputs using bounded enforcement"""
    print(f"\n{name}:")
    print("-" * 80)
    
    # Create all states
    state_objects = {name: state(name) for name in states}
    
    # Create trap state separately
    state_objects['T'] = state('T')
    # Make trap state loop to itself for all inputs
    for symbol in alphabet:
        state_objects['T'].transit[symbol] = state_objects['T']
    
    # Define transitions
    for (src, sym), dest in transition_map.items():
        state_objects[src].transit[sym] = state_objects[dest]
    
    # Define trap transitions (any undefined transition goes to a trap state)
    for s_name in list(state_objects.keys()):  # Create a copy of keys to iterate
        if s_name != 'T':  # Skip the trap state itself
            s = state_objects[s_name]
            for a in alphabet:
                if a not in s.transit:
                    s.transit[a] = state_objects['T']
    
    # Construct DFA
    q0 = state_objects[start_state]
    Q = list(state_objects.values())
    d = lambda state, a: state.transit[a] 
    F = lambda state: state.name in accept_states
    
    # Create the DFA object
    dfa = DFA(name, alphabet, Q, q0, F, d, [state_objects[s] for s in accept_states])
    
    # Test with each input sequence
    for test_name, input_seq in test_inputs.items():
        print(f"Testing: {test_name}")
        print(f"Input: {input_seq}")
        
        # Use bounded enforcer to process input
        enforced_output = enforcer(dfa, list(input_seq), max_buffer)
        
        print(f"Enforced Output: {''.join(enforced_output)}")
        print()
    
    return dfa

def create_reentrant_properties():
    """Create and return all reentrant application property DFAs"""
    
    # Common test inputs for all properties
    test_inputs = {
        "Valid Simple": "oooxo",
        "Valid Complex": "ooooxoooxo",
        "Invalid": "oooxooxoo"
    }
    
    # 1. No Reentrant Call in a Critical Section
    no_reentrant_call = build_and_test_property(
        name="No Reentrant Call in a Critical Section",
        alphabet=['e', 'x', 'o','l','u','c','r','i','j'],
        states=['q0', 'q1', 'q2','T'],
        transition_map={
            ('q0', 'o'): 'q0',
            ('q0', 'l'): 'q0',
            ('q0', 'u'): 'q0',
            ('q0', 'c'): 'q0',
            ('q0', 'r'): 'q0',
            ('q0', 'i'): 'q0',
            ('q0', 'j'): 'q0',
            ('q0', 'e'): 'q1',
            ('q0', 'x'): 'T',
            ('q1', 'o'): 'q1',
            ('q1', 'l'): 'q1',
            ('q1', 'u'): 'q1',
            ('q1', 'c'): 'q1',
            ('q1', 'r'): 'q1',
            ('q1', 'i'): 'q1',
            ('q1', 'j'): 'q1',
            ('q1', 'x'): 'q2',
            ('q1', 'e'): 'T',
            ('q2', 'o'): 'q2',
            ('q2', 'l'): 'q2',
            ('q2', 'r'): 'q2',
            ('q2', 'c'): 'q2',
            ('q2', 'u'): 'q2',
            ('q2', 'i'): 'q2',
            ('q2', 'j'): 'q2',
            ('q2', 'x'): 'T',
            ('q2', 'e'): 'q1',
            ('T', 'o'): 'T',
            ('T', 'l'): 'T',
            ('T', 'u'): 'T',
            ('T', 'c'): 'T',
            ('T', 'r'): 'T',
            ('T', 'i'): 'T',
            ('T', 'j'): 'T',
            ('T', 'x'): 'T',
            ('T', 'e'): 'T'  # Violation - reentrant call
        },
        start_state='q0',
        accept_states=['q0','q2'],
        test_inputs={
            "Valid Sequence": "ooeoxoeoxo",
            "Violation (Reentrant Call)": "ooeeoexo",
            "Empty Critical Section": "oexoexo"
        }
    )
    
    # 2. Eventual Completion of Interrupt Handling
    interrupt_handling = build_and_test_property(
        name="Eventual Completion of Interrupt Handling",
        alphabet=['e', 'x', 'o','l','u','c','r','i','j'],
        states=['q0', 'q1', 'q2', 'T'],
        transition_map={
            ('q0', 'o'): 'q0',
            ('q0', 'l'): 'q0',
            ('q0', 'u'): 'q0',
            ('q0', 'c'): 'q0',
            ('q0', 'r'): 'q0',
            ('q0', 'e'): 'q0',
            ('q0', 'x'): 'q0',
            ('q0', 'i'): 'q1',
            ('q0', 'j'): 'T',
            ('q1', 'o'): 'q1',
            ('q1', 'l'): 'q1',
            ('q1', 'u'): 'q1',
            ('q1', 'c'): 'q1',
            ('q1', 'r'): 'q1',
            ('q1', 'e'): 'q1',
            ('q1', 'x'): 'q1',
            ('q1', 'j'): 'q2',
            ('q1', 'i'): 'T',
            ('q2', 'o'): 'q2',
            ('q2', 'l'): 'q2',
            ('q2', 'u'): 'q2',
            ('q2', 'c'): 'q2',
            ('q2', 'r'): 'q2',
            ('q2', 'e'): 'q2',
            ('q2', 'x'): 'q2',
            ('q2', 'j'): 'T',
            ('q2', 'i'): 'q1',
            ('T', 'o'): 'T',
            ('T', 'l'): 'T',
            ('T', 'u'): 'T',
            ('T', 'c'): 'T',
            ('T', 'r'): 'T',
            ('T', 'e'): 'T',
            ('T', 'x'): 'T',
            ('T', 'j'): 'T',
            ('T', 'i'): 'T'
        },
        start_state='q0',
        accept_states=['q0','q2'],
        test_inputs={
            "Valid Sequence": "ooiojioojo",
            "Violation (Missing End)": "ooioio",
            "Violation (Nested Interrupt)": "oiiojo"
        }
    )
    
    # 3. Bounded Depth of Reentrant Calls (Max Depth = 2)
    bounded_reentrant_calls = build_and_test_property(
        name="Bounded Depth of Reentrant Calls (Max Depth = 2)",
        alphabet=['e', 'x', 'o','l','u','c','r','i','j'],
        states=['q0', 'q1', 'q2', 'q3','q4','T'],
        transition_map={
            ('q0', 'o'): 'q0',
            ('q0', 'l'): 'q0',
            ('q0', 'u'): 'q0',
            ('q0', 'i'): 'q0',
            ('q0', 'j'): 'q0',
            ('q0', 'e'): 'q0',
            ('q0', 'x'): 'q0',
            ('q0', 'c'): 'q1',
            ('q0', 'r'): 'T',  
            ('q1', 'o'): 'q1',
            ('q1', 'l'): 'q1',
            ('q1', 'u'): 'q1',
            ('q1', 'i'): 'q1',
            ('q1', 'j'): 'q1',
            ('q1', 'e'): 'q1',
            ('q1', 'x'): 'q1',
            ('q1', 'c'): 'q2',
            ('q1', 'r'): 'q4',
            ('q2', 'o'): 'q2',
            ('q2', 'l'): 'q2',
            ('q2', 'u'): 'q2',
            ('q2', 'i'): 'q2',
            ('q2', 'j'): 'q2',
            ('q2', 'e'): 'q2',
            ('q2', 'x'): 'q2',
            ('q2', 'c'): 'T',  
            ('q2', 'r'): 'q3',
            ('q3', 'o'): 'q3',
            ('q3', 'l'): 'q3',
            ('q3', 'u'): 'q3',
            ('q3', 'i'): 'q3',
            ('q3', 'j'): 'q3',
            ('q3', 'e'): 'q3',
            ('q3', 'x'): 'q3',
            ('q3', 'c'): 'T',  
            ('q3', 'r'): 'q4',
            ('q4', 'o'): 'q4',
            ('q4', 'l'): 'q4',
            ('q4', 'u'): 'q4',
            ('q4', 'i'): 'q4',
            ('q4', 'j'): 'q4',
            ('q4', 'e'): 'q4',
            ('q4', 'x'): 'q4',
            ('q4', 'c'): 'q1',  
            ('q4', 'r'): 'T',
            ('T', 'o'): 'T',
            ('T', 'i'): 'T',
            ('T', 'j'): 'T',
            ('T', 'l'): 'T',
            ('T', 'u'): 'T',
            ('T', 'e'): 'T',
            ('T', 'x'): 'T',
            ('T', 'c'): 'T',
            ('T', 'r'): 'T'
        },
        start_state='q0',
        accept_states=['q0','q4'],
        test_inputs={
            "Valid Sequence (Depth 1)": "ocoro",
            "Valid Sequence (Depth 2)": "occoror",
            "Violation (Depth 3)": "occocorro",
            "Mixed Valid Sequence": "ocoroccororo"
        }
    )
    
    # 4. No Premature Return
    no_premature_return = build_and_test_property(
        name="No Premature Return",
        alphabet=['e', 'x', 'o','l','u','c','r','i','j'],
        states=['q0', 'q1','q2','T'],
        transition_map={
            ('q0', 'o'): 'q0',
            ('q0', 'l'): 'q0',
            ('q0', 'u'): 'q0',
            ('q0', 'i'): 'q0',
            ('q0', 'j'): 'q0',
            ('q0', 'e'): 'q0',
            ('q0', 'x'): 'q0',
            ('q0', 'c'): 'q1',
            ('q0', 'r'): 'T',
            ('q1', 'o'): 'q1',
            ('q1', 'l'): 'q1',
            ('q1', 'u'): 'q1',
            ('q1', 'i'): 'q1',
            ('q1', 'j'): 'q1',
            ('q1', 'e'): 'q1',
            ('q1', 'x'): 'q1',
            ('q1', 'r'): 'q2',
            ('q1', 'c'): 'T',
             ('q2', 'o'): 'q2',
            ('q2', 'l'): 'q2',
            ('q2', 'u'): 'q2',
            ('q2', 'i'): 'q2',
            ('q2', 'j'): 'q2',
            ('q2', 'e'): 'q2',
            ('q2', 'x'): 'q2',
            ('q2', 'r'): 'T',
            ('q2', 'c'): 'q1',
            ('T', 'o'): 'T',
            ('T', 'i'): 'T',
            ('T', 'j'): 'T',
            ('T', 'l'): 'T',
            ('T', 'u'): 'T',
            ('T', 'e'): 'T',
            ('T', 'x'): 'T',
            ('T', 'r'): 'T',
            ('T', 'c'): 'T' 
        },
        start_state='q0',
        accept_states=['q0','q2'],
        test_inputs={
            "Valid Sequence": "oocorcooro",
            "Violation (Premature Return)": "oorcocor",
            "Balanced Calls": "ocrcocro"
        }
    )
    
    # 5. Non-Reentrant Locking Protocol
    non_reentrant_locking = build_and_test_property(
        name="Non-Reentrant Locking Protocol",
        alphabet=['e', 'x', 'o','l','u','c','r','i','j'],
        states=['q0', 'q1','q2','T'],
        transition_map={
           ('q0', 'o'): 'q0',
           ('q0', 'i'): 'q0',
           ('q0', 'j'): 'q0',
           ('q0', 'c'): 'q0',
           ('q0', 'r'): 'q0',
           ('q0', 'e'): 'q0',
           ('q0', 'x'): 'q0',
            ('q0', 'l'): 'q1',
            ('q0', 'u'): 'T',
            ('q1', 'o'): 'q1',
           ('q1', 'i'): 'q1',
           ('q1', 'j'): 'q1',
           ('q1', 'c'): 'q1',
           ('q1', 'r'): 'q1',
           ('q1', 'e'): 'q1',
           ('q1', 'x'): 'q1',
            ('q1', 'u'): 'q2',
            ('q1', 'l'): 'T',
            ('q2', 'o'): 'q2',
           ('q2', 'i'): 'q2',
           ('q2', 'j'): 'q2',
           ('q2', 'c'): 'q2',
           ('q2', 'r'): 'q2',
           ('q2', 'e'): 'q2',
           ('q2', 'x'): 'q2',
            ('q2', 'u'): 'T',
            ('q2', 'l'): 'q1',
            ('T', 'o'): 'T',
            ('T', 'i'): 'T',
            ('T', 'j'): 'T',
            ('T', 'l'): 'T',
            ('T', 'u'): 'T',
            ('T', 'e'): 'T',
            ('T', 'x'): 'T',
            ('T', 'r'): 'T',
            ('T', 'c'): 'T'  
        },
        start_state='q0',
        accept_states=['q0','q2'],
        test_inputs={
            "Valid Sequence": "ooloulou",
            "Violation (Nested Lock)": "ollouo",
            "Empty Critical Section": "oluo"
        }
    )
    
    return {
        "no_reentrant_call": no_reentrant_call,
        "interrupt_handling": interrupt_handling,
        "bounded_reentrant_calls": bounded_reentrant_calls,
        "no_premature_return": no_premature_return,
        "non_reentrant_locking": non_reentrant_locking
    }


def test_monolithic_intersection_enforcement(properties):
    """
    Tests monolithic enforcer created from the intersection of all five properties.
    Uses a complex input string containing all alphabet characters.
    """
    print("\nTesting Monolithic Intersection of All Properties:")
    print("=" * 80)
    
    # Create a monolithic enforcer combining all five properties
    all_properties = list(properties.values())
    monolithic = monolithic_enforcer("All_Properties", *all_properties)
    
    # Create a complex test input with all alphabet symbols
    complex_input = "ooeoxocrijocroluoeoexoccroijolucrocr"
    print(f"Input: {complex_input}")
    
    # Test with different buffer sizes
    for buffer_size in [20]:
        # Use the enforcer function from enforcer.py
        enforced = enforcer(monolithic, list(complex_input), buffer_size)
        print(f"Buffer size {buffer_size}: {''.join(enforced)}")

def parallel_compositional_enforcer(properties, input_events, max_buffer=20):
    """
    Implements a simple parallel compositional enforcer without intelligent routing.
    All properties process the input independently, and their outputs are combined
    using the maximal substring approach from enforcer.py.
    
    Args:
        properties: Dictionary mapping property names to their DFA objects
        input_events: List of input events to process
        max_buffer: Maximum buffer size for enforcement
        
    Returns:
        String representing the enforced output sequence after taking maximal substring
    """
    from src.enforcer import longest_common_subsequence
    
    print("\nSimple Parallel Compositional Enforcement:")
    print("-" * 80)
    print(f"Input: {''.join(input_events)}")
    
    # Process the input with each property independently
    individual_outputs = {}
    for name, dfa in properties.items():
        # Use the enforcer function to process the entire input
        enforced_output = enforcer(dfa, input_events, max_buffer)
        individual_outputs[name] = ''.join(enforced_output)
        print(f"  {name}: {individual_outputs[name]}")
    
    # Compute the longest common subsequence (maximal substring)
    # Start with the output of the first property
    if not individual_outputs:
        return ""
        
    outputs = list(individual_outputs.values())
    result = outputs[0]
    
    # Compute LCS with each subsequent property's output
    for output in outputs[1:]:
        result = longest_common_subsequence(result, output)
    
    print(f"Maximal substring result: {result}")
    return result

def test_individual_vs_combined_enforcement(properties):
    """
    Tests and compares individual enforcement vs. monolithic vs. parallel compositional enforcement.
    Uses the same complex input for all tests to allow fair comparison.
    """
    print("\nComparing Individual vs. Monolithic vs. Parallel Enforcement:")
    print("=" * 80)
    
    # Create a complex test input with all alphabet symbols
    complex_input = "ooeoxocrijocroluoeoexoccroijolucrocr"
    print(f"Input: {complex_input}")
    
    # 1. Test individual property enforcement
    print("\n1. Individual Property Enforcement:")
    individual_results = {}
    for name, dfa in properties.items():
        # Use standard buffer size of 20 for all tests
        enforced = enforcer(dfa, list(complex_input), 20)
        individual_results[name] = ''.join(enforced)
        print(f"  {name}: {individual_results[name]}")
    
    # 2. Test monolithic enforcement (using all properties)
    print("\n2. Monolithic Enforcement (All Properties):")
    all_props = list(properties.values())
    monolithic = monolithic_enforcer("All_Properties", *all_props)
    try:
        monolithic_result = enforcer(monolithic, list(complex_input), 20)
        print(f"  Result: {''.join(monolithic_result)}")
    except Exception as e:
        print(f"  Failed: {str(e)}")
    
    # 3. Test simple parallel compositional enforcement (based on maximal substring)
    print("\n3. Simple Parallel Compositional Enforcement:")
    parallel_result = parallel_compositional_enforcer(properties, list(complex_input), 20)
    print(f"  Result: {parallel_result}")
    
    # 4. Analysis of results
    print("\n4. Analysis of Results:")
    print("  Character counts in outputs:")
    print(f"    Original input: {len(complex_input)} characters")
    
    # Count individual results
    for name, result in individual_results.items():
        print(f"    {name}: {len(result)} characters")
    
    # Count monolithic result if available
    if 'monolithic_result' in locals():
        print(f"    Monolithic: {len(''.join(monolithic_result))} characters")
    
    # Count parallel result
    print(f"    Parallel: {len(parallel_result)} characters")

def comparison_analysis(properties):
    """
    Performs a detailed analysis of the differences between enforcement approaches.
    Shows which events are preserved/suppressed by each approach.
    """
    print("\nDetailed Comparison Analysis:")
    print("=" * 80)
    
    # A complex input with a strategic mix of symbols to test enforcement behavior
    comparison_input = "ocreluxijocrexoijolocr"
    print(f"Input: {comparison_input}")
    
    # Store results from each approach
    results = {}
    
    # 1. Test each individual property
    for name, dfa in properties.items():
        results[name] = ''.join(enforcer(dfa, list(comparison_input), 20))
    
    # 2. Test monolithic enforcement
    all_props = list(properties.values())
    monolithic = monolithic_enforcer("Monolithic", *all_props)
    try:
        results["monolithic"] = ''.join(enforcer(monolithic, list(comparison_input), 20))
    except Exception as e:
        results["monolithic"] = f"ERROR: {str(e)}"
    
    # 3. Test simple parallel compositional enforcement
    results["parallel"] = parallel_compositional_enforcer(properties, list(comparison_input), 20)
    
    # Create a comparison table
    print("\nEvent-by-event enforcement comparison:")
    print("-" * 80)
    print(f"{'Input':<8} | ", end="")
    for method in results.keys():
        print(f"{method:<15} | ", end="")
    print()
    print("-" * (8 + sum(len(k) + 17 for k in results.keys())))
    
    # Compare each character position
    for i, char in enumerate(comparison_input):
        print(f"{i}:{char:<6} | ", end="")
        for method, output in results.items():
            # Check if this character position exists in the output
            if i < len(output):
                if output[i] == char:
                    status = f"✓ {output[i]}"
                else:
                    status = f"? {output[i]}"
            else:
                status = "✗ (suppressed)"
            print(f"{status:<15} | ", end="")
        print()

def main():
    # Create all the reentrant application property DFAs
    properties = create_reentrant_properties()
    
    # Test monolithic intersection of all properties
    test_monolithic_intersection_enforcement(properties)
    
    # Test individual vs. combined enforcement approaches
    test_individual_vs_combined_enforcement(properties)

if __name__ == "__main__":
    main()