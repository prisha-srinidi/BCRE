import sys
import os
import time
import random
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.enforcer import state, DFA, enforcer, monolithic_enforcer, longest_common_subsequence, serial_enforcer



#The properties'automata are as follows".....................................................................................
def no_reentrant_call_before_update(alphabet):
    """
    Creates a DFA for "P1: No reentrant call before state update" property
    
    States:
    - q0: Initial state
    - q1: After first 'w' (waiting for update 'u')
    - q2: After proper update 'u' (accepting state)
    - q3: Error state (reentrant call occurred before update)
    - dead: Trap state for additional clarity
    """
    q0 = state('q0')     
    q1 = state('q1')     
    q2 = state('q2')     
    trap = state('T')     
    
    for a in alphabet:
        if a == 'w':
            q0.transit[a] = q1   
        elif a in ['u', 'r', 'c']:
            q0.transit[a] = q0    
    
        if a == 'w':
            q1.transit[a] = trap    
        elif a == 'u':
            q1.transit[a] = q2    
        elif a in ['r', 'c']:
            q1.transit[a] = q1   
    
        q2.transit[a] = q2       
         
        trap.transit[a] = trap       
    
    d = lambda state, a: state.transit[a]
    F = lambda state: state.name == 'q2'  
    
    return DFA('no_reentrant_call_before_update', alphabet, 
               [q0, q1, q2, trap], q0, F, d, [q2])

def in_one_section_only_one_invocation_of_critical_section(alphabet):
    """
    Creates a DFA for "P2: Single Invocation (Mutex Enforcement): In 1 section, only 1 invocation of critical section is allowed" property
    
    States:
    - q0: Initial state (outside critical section)
    - q1: Inside critical section
    - q2: After proper completion (accepting state)
    - q3: Error state (reentrant call occurred inside critical section)
    - T: Trap state
    """
    # States
    q0 = state('q0')    
    q1 = state('q1')    
    trap = state('T')     
    
    
    for a in alphabet:
        if a == 'w':
            q0.transit[a] = q0    
        elif a == 'u':
            q0.transit[a] = q0    
        elif a == 'r':
            q0.transit[a] = q0    
        elif a == 'c':
            q0.transit[a] = q1    
    
        if a == 'w':
            q1.transit[a] = q1    
        elif a == 'u':
            q1.transit[a] = q1    
        elif a == 'r':
            q1.transit[a] = q1    
        elif a == 'c':
            q1.transit[a] = trap    
    
        trap.transit[a] = trap        
            
    
    d = lambda state, a: state.transit[a]
    F = lambda state: state.name in ['q0','q1'] 
    
    return DFA('in_one_section_only_one_invocation_of_critical_section', alphabet, 
               [q0, q1, trap], q0, F, d, [q0, q1])

def no_reentrant_call_in_critical_section(alphabet):
    """
    Creates a DFA for "P3: No Reentrant Call in a Critical Section" property
    
    States:
    - q0: Initial state (outside critical section)
    - q1: Inside critical section
    - q2: After proper completion (accepting state)
    - q3: Error state (reentrant call occurred inside critical section)
    - T: Trap state
    """
    # States
    q0 = state('q0')     
    q1 = state('q1')     
    q2 = state('q2')    
    trap = state('T')     
    
    
    for a in alphabet:
        if a == 'w':
            q0.transit[a] = q0   
        elif a == 'u':
            q0.transit[a] = q0    
        elif a == 'r':
            q0.transit[a] = q0    
        elif a == 'c':
            q0.transit[a] = q1   
    
        if a == 'w':
            q1.transit[a] = trap   
        elif a == 'u':
            q1.transit[a] = q2   
        elif a == 'r':
            q1.transit[a] = q1   
        elif a == 'c':
            q1.transit[a] = q1    
    
        q2.transit[a] = q2        
    
        trap.transit[a] = trap        
    
    d = lambda state, a: state.transit[a]
    F = lambda state: state.name == 'q2'  
    
    return DFA('no_reentrant_call_in_critical_section', alphabet, 
               [q0, q1, q2, trap], q0, F, d, [q2])

def limit_state_update_to_three_in_one_session(alphabet):
    """
    Creates a DFA for "P4: Limiting the number of state updates: In 1 session, the maximum number of state updates allowed is 3." property
    
    States:
    - q0: Initial state (outside critical section)
    - q1: Inside critical section
    - q2: After proper completion (accepting state)
    - q3: Error state (reentrant call occurred inside critical section)
    - T: Trap state
    """
    q0 = state('q0')     
    q1 = state('q1')     
    q2 = state('q2')    
    q3 = state('q3')    
    trap = state('T')     
    
    
    for a in alphabet:
        if a == 'u':
            q0.transit[a] = q1    
        elif a == 'w':
            q0.transit[a] = q0   
        elif a == 'r':
            q0.transit[a] = q0   
        elif a == 'c':
            q0.transit[a] = q0   
    
        if a == 'u':
            q1.transit[a] = q2    
        elif a == 'w':
            q1.transit[a] = q1    
        elif a == 'r':
            q1.transit[a] = q1    
        elif a == 'c':
            q1.transit[a] = q1    
        if a == 'u':
            q2.transit[a] = q3   
        elif a == 'w':
            q2.transit[a] = q2    
        elif a == 'r':
            q2.transit[a] = q2    
        elif a == 'c':
            q2.transit[a] = q2    
        if a == 'u':
            q3.transit[a] = trap    
        elif a == 'w':
            q3.transit[a] = q3    
        elif a == 'r':
            q3.transit[a] = q3    
        elif a == 'c':
            q3.transit[a] = q3        
        trap.transit[a] = trap        
            
    
    d = lambda state, a: state.transit[a]
    F = lambda state: state.name in ['q0','q1','q2','q3'] 
    
    return DFA('limit_state_update_to_three_in_one_session', alphabet, 
               [q0, q1, q2, q3, trap], q0, F, d, [q0, q1,q2,q3])

def no_access_after_release(alphabet):
    """
    Creates a DFA for "P5: No Access After Release: After being released, resources cannot be accessed in the critical section.
    
    States:
    - q0: Initial state (outside critical section)
    - q1: Inside critical section
    - q2: After proper completion (accepting state)
    - q3: Error state (reentrant call occurred inside critical section)
    - T: Trap state
    """
    q0 = state('q0')     
    q1 = state('q1') 
    q2 = state('q2')     
    trap = state('T')     
    
    
    for a in alphabet:
        if a == 'w':
            q0.transit[a] = q0    
        elif a == 'u':
            q0.transit[a] = q0   
        elif a == 'r':
            q0.transit[a] = q1    
        elif a == 'c':
            q0.transit[a] = q0   
    
        if a == 'w':
            q1.transit[a] = q2   
        elif a == 'u':
            q1.transit[a] = trap   
        elif a == 'r':
            q1.transit[a] = q2    
        elif a == 'c':
            q1.transit[a] = trap   
        
        q2.transit[a] = q2      
    
        trap.transit[a] = trap       
            
    
    d = lambda state, a: state.transit[a]
    F = lambda state: state.name in ['q0','q1','q2'] 
    
    return DFA('no_access_after_release', alphabet, 
               [q0, q1, q2, trap], q0, F, d, [q0, q1, q2])

#The properties'automata modelling is done.....................................................................................





def measure_enforcer_performance(dfa, input_string, buffer_size=10, iterations=1):
    """Measure the performance of the enforcer function"""
    enforce_times = []
    print("single enforcer")
    
    for _ in range(iterations):
        start_time = time.time()
        enforced_output = enforcer(dfa, list(input_string), buffer_size)
        end_time = time.time()
        enforce_times.append(end_time - start_time)
    
    return {
        'mean': np.mean(enforce_times),
        'median': np.median(enforce_times),
        'min': np.min(enforce_times),
        'max': np.max(enforce_times),
        'output_length': len(''.join(enforced_output)),
        'input_length': len(input_string)
    }

def measure_monolithic_performance(dfa1, dfa2, dfa3, dfa4, dfa5, input_string, buffer_size=10, iterations=1):
    enforce_times = []
    print("monolithic")
    
    monolithic = monolithic_enforcer("combined", dfa1, dfa2, dfa3, dfa4, dfa5)
    
    for _ in range(iterations):
        if hasattr(monolithic, 'reset'):
            monolithic.reset()
        
        if hasattr(monolithic, 'buffer'):
            monolithic.buffer = []
        
        start_time = time.time()
        enforced_output = enforcer(monolithic, list(input_string), buffer_size)
        end_time = time.time()
        enforce_times.append(end_time - start_time)
    
    return {
        'mean': np.mean(enforce_times),
        'median': np.median(enforce_times),
        'min': np.min(enforce_times),
        'max': np.max(enforce_times),
        'output_length': len(''.join(enforced_output)),
        'input_length': len(input_string)
    }
    
"""
def measure_parallel_lcs_performance(dfa1, dfa2, dfa3, dfa4, dfa5, input_string, buffer_size=10, iterations=1):
    #Measure the performance of parallel LCS enforcement with separate timing for enforcement and LCS
    enforcement_times = []
    lcs_times = []
    print("parallel")
    
    lcs_output = None 
    
    for _ in range(iterations):
        start_time1 = time.time()
        output1 = enforcer(dfa1, list(input_string), buffer_size)
        output2 = enforcer(dfa2, list(input_string), buffer_size)
        output3 = enforcer(dfa3, list(input_string), buffer_size)
        output4 = enforcer(dfa4, list(input_string), buffer_size)
        output5 = enforcer(dfa5, list(input_string), buffer_size)
        end_time1 = time.time()
        enforcement_time = end_time1 - start_time1
        enforcement_times.append(enforcement_time)
        
        start_time2 = time.time()
        lcs_output = longest_common_subsequence(''.join(output1), ''.join(output2))
        lcs_output = longest_common_subsequence(lcs_output, ''.join(output3))
        lcs_output = longest_common_subsequence(lcs_output, ''.join(output4))
        lcs_output = longest_common_subsequence(lcs_output, ''.join(output5))
        end_time2 = time.time()
        lcs_time = end_time2 - start_time2
        lcs_times.append(lcs_time)
    
    total_times = [enforcement + lcs for enforcement, lcs in zip(enforcement_times, lcs_times)]
    
    return {
        'mean': np.mean(total_times),                   
        'mean_enforcement': np.mean(enforcement_times),
        'mean_lcs': np.mean(lcs_times),                 
        'median': np.median(total_times),
        'min': np.min(total_times),
        'max': np.max(total_times),
        'output_length': len(lcs_output),
        'input_length': len(input_string)
    }
"""

def measure_serial_performance(dfa1, dfa2,dfa3,dfa4,dfa5, input_string, buffer_size=10, iterations=1):
    """Measure the performance of serial enforcement"""
    enforce_times = []
    print("serial")
    
    serial = serial_enforcer("serial", dfa1, dfa2,dfa3,dfa4,dfa5)
    
    for _ in range(iterations):
        start_time = time.time()
        enforced_output, _ = serial(list(input_string), buffer_size)
        end_time = time.time()
        enforce_times.append(end_time - start_time)
    
    return {
        'mean': np.mean(enforce_times),
        'median': np.median(enforce_times),
        'min': np.min(enforce_times),
        'max': np.max(enforce_times),
        'output_length': len(''.join(enforced_output)),
        'input_length': len(input_string)
    }

def generate_random_string(alphabet, length):
    """Generate a random string of given length from the given alphabet with specific pattern:
    - Starts with characters from 'wcr'
    - Ends with 'u'
    - Middle portion is random
    """
    if length < 3:
        return 'w' + 'u'
    
    
    main_length = length 
    
    main_chars = ['w', 'c', 'r','u']
    main_part = ''.join(random.choice(main_chars) for _ in range(main_length))
    
    return main_part

def main():
    shared_alphabet = ['w','c','r','u']
    
    dfa1 = no_reentrant_call_before_update(shared_alphabet)
    dfa2= in_one_section_only_one_invocation_of_critical_section(shared_alphabet)
    dfa3 = no_reentrant_call_in_critical_section(shared_alphabet)
    dfa4 = limit_state_update_to_three_in_one_session(shared_alphabet)
    dfa5 = no_access_after_release(shared_alphabet)

    
    input_sizes = [10,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    
    results = pd.DataFrame()
    
    for size in input_sizes:
        print(f"Testing with input size {size}...")
        
        input_string = generate_random_string(shared_alphabet, size)
        print(f"String starts with: {input_string[:5]}... ends with: ...{input_string[-5:]}")
        
        buffer_size = 10
        
        enforcer_perf = measure_enforcer_performance(dfa1, input_string, buffer_size=buffer_size)
        enforcer_perf = measure_enforcer_performance(dfa2, input_string, buffer_size=buffer_size)
        enforcer_perf = measure_enforcer_performance(dfa3, input_string, buffer_size=buffer_size)
        enforcer_perf = measure_enforcer_performance(dfa4, input_string, buffer_size=buffer_size)
        enforcer_perf = measure_enforcer_performance(dfa5, input_string, buffer_size=buffer_size)
        monolithic_perf = measure_monolithic_performance(dfa1, dfa2, dfa3, dfa4, dfa5, input_string, buffer_size=buffer_size)
        # parallel_perf = measure_parallel_lcs_performance(dfa1, dfa2, dfa3, dfa4, dfa5, input_string, buffer_size=buffer_size)
        serial_perf = measure_serial_performance(dfa1, dfa2, dfa3, dfa4, dfa5, input_string, buffer_size=buffer_size)
        
        result_row = {
            'input_size': size,
            'single_enforcer_mean_time': enforcer_perf['mean'],
            'single_enforcer_output_length': enforcer_perf['output_length'],
            'monolithic_mean_time': monolithic_perf['mean'],
            'monolithic_output_length': monolithic_perf['output_length'],
            # 'parallel_lcs_mean_time': parallel_perf['mean'],
            # 'parallel_enforcement_time': parallel_perf['mean_enforcement'],
            # 'parallel_lcs_calc_time': parallel_perf['mean_lcs'],
            # 'parallel_lcs_output_length': parallel_perf['output_length'],
            'serial_mean_time': serial_perf['mean'],
            'serial_output_length': serial_perf['output_length']
        }
        
        results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)
        
        print(f"Single Enforcer: {enforcer_perf['mean']:.6f} seconds")
        print(f"Monolithic: {monolithic_perf['mean']:.6f} seconds")
        # print(f"Parallel LCS: {parallel_perf['mean']:.6f} seconds")
        print(f"Serial: {serial_perf['mean']:.6f} seconds")
        # print(f"Output lengths: {enforcer_perf['output_length']}, {monolithic_perf['output_length']}, {parallel_perf['output_length']}, {serial_perf['output_length']}")
        print("-" * 50)
    
    results.to_csv('enforcer_performance_comparison.csv', index=False)
    
    print("\nPerformance Summary:")
    print("=" * 50)
    print(results)
    
    results['mono_to_single_ratio'] = results['monolithic_mean_time'] / results['single_enforcer_mean_time']
    # results['parallel_to_mono_ratio'] = results['parallel_lcs_mean_time'] / results['monolithic_mean_time']
    results['serial_to_mono_ratio'] = results['serial_mean_time'] / results['monolithic_mean_time']
    
    print("\nPerformance Ratios:")
    print("=" * 50)
    print(f"Monolithic/Single: {results['mono_to_single_ratio'].mean():.2f}x")
    # print(f"Parallel/Monolithic: {results['parallel_to_mono_ratio'].mean():.2f}x")
    print(f"Serial/Monolithic: {results['serial_to_mono_ratio'].mean():.2f}x")

if __name__ == "__main__":
    main()
