# Bounded Compositional Runtime Enforcement

This repository implements a framework for monitoring and enforcing correctness properties on reentrant applications using deterministic finite automata (DFA). The codebase is organized into two main directories: `src` and `casestudy`.

---

## 📁 Directory Structure

### 1. `src/`
This directory contains the core implementation of automata operations and enforcer logic.

#### • `automata.py`
Implements basic operations on automata:
- Transition creation
- Reset operations
- Other automaton manipulations

#### • `enforcer.py`
Implements both **bounded** and **compositional enforcers** for runtime enforcement of properties.

Key Functions:
- **`computeEmptinessDict`**: Precomputes emptiness for each state in the automaton. Returns a dictionary mapping states to a boolean — `True` if the state is dead (empty language), `False` otherwise.
- **`computes_substring`**: Extracts a substring from the buffer by removing the smallest repeating cycle.
- **`Clean`**: Cleans the buffer using the substring computed by `computes_substring`.
- **`Enforcer`**: A bounded memory enforcer that incrementally produces an output sequence satisfying the property.

Compositional Enforcement Functions:
- **`product`**: Constructs the product of two DFAs.
- **`monolithic_enforcer`**: Builds a monolithic enforcer by composing multiple DFAs via product construction.
- **`serial_enforcer`**: Implements serial composition of multiple enforcers.

---

### 2. `casestudy/`
Contains performance evaluation scripts and DFA definitions for a reentrant application case study.

#### • `performance.py`
Defines five DFAs representing the following safety properties:
1. `no_reentrant_call_before_update`
2. `in_one_section_only_one_invocation_of_critical_section`
3. `no_reentrant_call_in_critical_section`
4. `limit_state_update_to_three_in_one_session`
5. `no_access_after_release`

Also includes performance measurement functions:
- `measure_monolithic_performance`
- `measure_serial_performance`

Trace Generation:
- `generate_random_string`: Generates random traces used for evaluating enforcer performance.

---

## 🛠️ Usage

1. Navigate to the project root.
2. Ensure all required Python dependencies are installed.
3. Run case studies or evaluate performance using scripts in `casestudy/performance.py`.

---

## 📌 Notes

- The project uses deterministic finite automata (DFA) to enforce properties at runtime.
- Supports both monolithic (via product automata) and serial composition of enforcers.

---

## 📂 Authors & Acknowledgments

This project was developed to evaluate performance of the "Bounded Serial Compositional Runtime Enforcement" framework.

---
