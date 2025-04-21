import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.enforcer import state, DFA, enforcer, monolithic_enforcer, longest_common_subsequence

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
    
    # 1. No reentrant call before state update 
    state_update = build_and_test_property(
        name="No reentrant call before state update",
        alphabet=['w','u','c','r'],
        states=['q0', 'q1', 'q2','q3'],
        transition_map={
            ('q0', 'w'): 'q1',
            ('q0', 'u'): 'q0',
            ('q0', 'r'): 'q0',
            ('q0', 'c'): 'q0',

            ('q1', 'w'): 'q3',
            ('q1', 'u'): 'q2',
            ('q1', 'r'): 'q1',
            ('q1', 'c'): 'q1',
            
            ('q2', 'w'): 'q2',
            ('q2', 'u'): 'q2',
            ('q2', 'r'): 'q2',
            ('q2', 'c'): 'q2',

            ('q3', 'w'): 'q3',
            ('q3', 'u'): 'q3',
            ('q3', 'r'): 'q3',
            ('q3', 'c'): 'q3',
        },
        start_state='q0',
        accept_states=['q2'],
        test_inputs={
            "Valid Sequence": "wcrwu",
            "Violation (Reentrant Call)": "wcw",
        }
    )
    
    # 2. In one session, only one invocation of critical section 
    cs_once = build_and_test_property(
        name="In one session, only one invocation of critical section ",
        alphabet=['w','u','c','r'],
        states=['q0', 'q1', 'q2'],
        transition_map={
            ('q0', 'w'): 'q0',
            ('q0', 'u'): 'q0',
            ('q0', 'r'): 'q0',
            ('q0', 'c'): 'q1',

            ('q1', 'w'): 'q1',
            ('q1', 'u'): 'q1',
            ('q1', 'r'): 'q1',
            ('q1', 'c'): 'q2',
            
            ('q2', 'w'): 'q2',
            ('q2', 'u'): 'q2',
            ('q2', 'r'): 'q2',
            ('q2', 'c'): 'q2',
        },
        start_state='q0',
        accept_states=['q0','q1'],
        test_inputs={
            "Valid Sequence": "ooeoxoeoxo",
            "Violation (Reentrant Call)": "ooeeoexo",
            "Empty Critical Section": "oexoexo"
        }
    )
    # 3. No Reentrant Call in a Critical Section
    no_reentrant_call = build_and_test_property(
        name="No Reentrant Call in a Critical Section",
        alphabet=['w','u','c','r'],
        states=['q0', 'q1', 'q2','q3'],
        transition_map={
            ('q0', 'w'): 'q0',
            ('q0', 'u'): 'q0',
            ('q0', 'r'): 'q0',
            ('q0', 'c'): 'q1',
            ('q1', 'w'): 'q3',
            ('q1', 'u'): 'q2',
            ('q1', 'r'): 'q1',
            ('q1', 'c'): 'q1',
            
            ('q2', 'w'): 'q2',
            ('q2', 'u'): 'q2',
            ('q2', 'r'): 'q2',
            ('q2', 'c'): 'q2',

            ('q3', 'w'): 'q3',
            ('q3', 'u'): 'q3',
            ('q3', 'r'): 'q3',
            ('q3', 'c'): 'q3',
        },
        start_state='q0',
        accept_states=['q2'],
        test_inputs={
            "Valid Sequence": "ooeoxoeoxo",
            "Violation (Reentrant Call)": "ooeeoexo",
            "Empty Critical Section": "oexoexo"
        }
    )
    
    
    # 4. In one session, maximum reentrant calls allowed=3
    max_reentrant_calls = build_and_test_property(
        name="In one session, maximum reentrant calls allowed=3",
        alphabet=['w','u','c','r'],
        states=['q0', 'q1', 'q2','q3','q4'],
        transition_map={
            ('q0', 'w'): 'q1',
            ('q0', 'u'): 'q0',
            ('q0', 'r'): 'q0',
            ('q0', 'c'): 'q0',

            ('q1', 'w'): 'q2',
            ('q1', 'u'): 'q1',
            ('q1', 'r'): 'q1',
            ('q1', 'c'): 'q1',
            
            ('q2', 'w'): 'q3',
            ('q2', 'u'): 'q2',
            ('q2', 'r'): 'q2',
            ('q2', 'c'): 'q2',

            ('q3', 'w'): 'q4',
            ('q3', 'u'): 'q3',
            ('q3', 'r'): 'q3',
            ('q3', 'c'): 'q3',

            ('q4', 'w'): 'q4',
            ('q4', 'u'): 'q4',
            ('q4', 'r'): 'q4',
            ('q4', 'c'): 'q4',
        },
        start_state='q0',
        accept_states=['q0','q1','q2','q3'],
        test_inputs={
            "Valid Sequence": "ooeoxoeoxo",
            "Violation (Reentrant Call)": "ooeeoexo",
            "Empty Critical Section": "oexoexo"
        }
    )
    
    # 5. After release, resource should not be acquired again in critical section
    release_acquire = build_and_test_property(
        name="After release, resource should not be acquired again in critical section",
        alphabet=['w','u','c','r'],
        states=['q0', 'q1', 'q2','q3'],
        transition_map={
            ('q0', 'w'): 'q0',
            ('q0', 'u'): 'q0',
            ('q0', 'r'): 'q1',
            ('q0', 'c'): 'q0',

            ('q1', 'w'): 'q2',
            ('q1', 'u'): 'q3',
            ('q1', 'r'): 'q2',
            ('q1', 'c'): 'q3',
            
            ('q2', 'w'): 'q2',
            ('q2', 'u'): 'q2',
            ('q2', 'r'): 'q2',
            ('q2', 'c'): 'q2',

            ('q3', 'w'): 'q3',
            ('q3', 'u'): 'q3',
            ('q3', 'r'): 'q3',
            ('q3', 'c'): 'q3',
        },
        start_state='q0',
        accept_states=['q0','q1','q2'],
        test_inputs={
            "Valid Sequence": "ooeoxoeoxo",
            "Violation (Reentrant Call)": "ooeeoexo",
            "Empty Critical Section": "oexoexo"
        }
    )
    
    return {
        "state_update": state_update,
        "cs_once": cs_once,
        "no_reentrant_call": no_reentrant_call,
        "max_reentrant_calls": max_reentrant_calls,
        "release_acquire": release_acquire
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
    complex_input = "wcrcrcrcrcrcrcwcrwcrwcrwcrwcrwcrwcrwcruuuu"
    print(f"Input: {complex_input}")
    
    # Test with different buffer sizes
    for buffer_size in [5]:
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
    complex_input = "wcrwcrwcrwcrwcrwcrwcrwcruuuuuu"
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